
# START 这是分类任务!!!!!!!!!
#当你安装好所有环境后，开始数据集处理与模型训练（微调）
# 2. Dataset Check 数据集

from datetime import date, datetime
import imp
import time
import re
import traceback
import pandas as pd
from datasets import load_dataset
import os
import numpy as np
from datasets import DatasetDict
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from torch.nn import DataParallel
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, LlamaForSequenceClassification
import evaluate
import json
from torch.nn.parallel import DistributedDataParallel

def time_shower(main):
    def call_main():
        print("\033[1;34m.....Main Start.....\033[0m")
        start = time.time()
        main()
        end = time.time()
        print("\n\033[1;34m.....Main End.....")
        print(f"-----共耗费了：{(end-start):.4f} 秒-----\033[0m")
    return call_main



def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)




@time_shower
def main():


    # Setting for A100 - For RTX 3090
    #超参数
    BASE_MODEL = "decapoda-research/llama-7b-hf"
    #BASE_MODEL = "meta-llama/Llama-2-7b-hf"
    output_dir = "./lora-alpaca-wmy-classify-7B-10ep"
    TRAIN_DIR = "./train_dataset/meld_train_joined_forClassification.json"
    # num_labels 非常重要，不同分类任务，类别数目是不同的！！！
    num_labels = 7  # 7分类，如果遇到其他任务，需要修改。

    MICRO_BATCH_SIZE = 4  # change to 4 for 3090，默认是4
    BATCH_SIZE = 48  # 为了加快训练，将默认的128加大为256，容易爆显存
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    # GRADIENT_ACCUMULATION_STEPS = 2
    EPOCHS = 10  # paper uses 3
    LEARNING_RATE = 2e-5  # from the original paper
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data 截断长度
    LORA_R = 4  # LORA中最重要的一个超参数，用于降维，LORA的秩
    LORA_ALPHA = 16  # alpha其实是个缩放参数，本质和learning rate相同，所以为了简化默认让alpha=rank
    LORA_DROPOUT = 0.05
    accuracy = evaluate.load("accuracy")

    # 1---分词器处理
    # 加载LLAMA-13B的分词器, from_pretrained:加载预训练模型
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL, add_eos_token=True)
    # 指定我们使用 [SEP] 标记来进行补充填充,而不是默认的 [PAD] 标记
    tokenizer.pad_token = tokenizer.eos_token
    # 特殊token [EOS]的位置。它被用来指示模型当前生成的句子已经结束
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2-----数据集处理与 划分训练集、验证集

    data = load_dataset("json", data_files=TRAIN_DIR)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=CUTOFF_LEN, padding="max_length")

    # 打乱数据，设置训练集,截断、最大长度是256，可以包容96的数据，也可以更大设置为512，但是会慢
    train_val = data["train"].train_test_split(
        test_size=3000, shuffle=True, seed=42
    )
    print(train_val)
    tokenized_train_val = train_val.map(preprocess_function, batched=True)

    train_data = tokenized_train_val["train"]
    val_data = tokenized_train_val["test"]

    print("The dataset has been loaded.......Next load model and train model...")
    print("-----------------------------------------")

    # 3---开始模型调用与lora合并

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    # LlamaForCausalLM，可用于文本生成;  LlamaForSequenceClassification，可用于文本分类

    id2label = {0: "NEUTRAL", 1: "ANGER", 2: "JOY", 3: "SADNESS", 4: "FEAR", 5: "DISGUST", 6: "SURPRISE"}
    label2id = {"NEUTRAL": 0, "ANGER": 1, "JOY": 2, "SADNESS": 3, "FEAR": 4, "DISGUST": 5, "SURPRISE": 6}

    model = LlamaForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=num_labels, id2label=id2label,
                                                           label2id=label2id, load_in_8bit=True, device_map=device_map)
    # prepare_model_for_int8_training 对在Lora微调中使用LLM.int8()进行了适配
    model = prepare_model_for_int8_training(model)
    print("peft...")

    # 设置一下LORA的config，调用一下get_peft_model方法，就获得了在原模型基础上的PEFT模型
    # config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=["q_proj", "v_proj"], lora_dropout=LORA_DROPOUT, bias="none", task_type="SEQ_CLS",)
    # model = get_peft_model(model, config) #获得了在原模型基础上的PEFT模型
    # tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token,补齐符号为0

    # 4----执行模型训练

    # model.print_trainable_parameters()  # Be more transparent about the % of trainable params

    print("now set the multi-GPU here...")  # 并行

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        model.find_unused_parameters = True

    trainer = transformers.Trainer(
        model=model,
        # train_dataset=data["train"], #已经建立好的训练集，如果没有需要自己提前分好训练集和测试集
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            per_device_eval_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            weight_decay=0.01,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            ddp_find_unused_parameters=False if ddp else None,
            output_dir=output_dir,  # 模型训练完的输出路径
            save_total_limit=3,
        ),
        # DataCollatorWithPadding 用于分类任务的数据规范。
        data_collator=transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    )
    model.config.use_cache = False
    print("Now training...")
    trainer.train(resume_from_checkpoint=False)  # resume_from_checkpoint 是否断点续训，开始训练
    print("....Model Train End....., Now, save the model")
    model.save_pretrained(output_dir)  # 保存:它只包含 2 个文件: adapter_config.json 和 adapter_model.bin是保存经过训练的增量 PEFT 权重

    # 5---未来可以考虑将自己的模型上传
    #后期可以将自己训练好的模型上传到huggingface上，步骤是：
    # from huggingface_hub import notebook_login

    # notebook_login()

    # model.push_to_hub("affectLLM/Sentiment", use_auth_token=True)




if __name__ == '__main__':
    main()
