
# START 【生成任务训练，加入训练集与验证集】
#当你安装好所有环境后，开始数据集处理与模型训练（微调）
# 2. Dataset Check 数据集

from datetime import date, datetime
import imp
import time
import traceback

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
import torch
import torch.nn as nn
from torch.nn import DataParallel
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

def time_shower(main):
    def call_main():
        print("\033[1;34m.....Main Start.....\033[0m")
        start = time.time()
        main()
        end = time.time()
        print("\n\033[1;34m.....Main End.....")
        print(f"-----共耗费了：{(end-start):.4f} 秒-----\033[0m")
    return call_main

#     ### Speaker:
#     {data_point["Speaker"]}
# ### Speaker:
#             {data_point["Speaker"]}

def generate_prompt(data_point):
    # create prompts from the loaded dataset and tokenize them
    if data_point["input"]:

        if "Context" not in data_point:
            return f"""Please perform Classification task. Given the sentence, assign the correct label. Return label only without any other text.
            ### Instruction:
            {data_point["instruction"]}

            ### Input:
            {data_point["input"]}

            ### Response:
            {data_point["output"]}"""
        else:
            return f"""Please perform Classification task. Given the sentence, assign the correct label. Return label only without any other text.
            ### Instruction:
            {data_point["instruction"]}

            ### Context:
            {data_point["Context"]}

            ### Input:
            {data_point["input"]}

            ### Response:
            {data_point["output"]}"""

    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {data_point["instruction"]}

        ### Response:
        {data_point["output"]}"""
#"decapoda-research/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
def tokenize(prompt, add_eos_token=True):

    #tokenizer.pad_token = tokenizer.eos_token
    # 特殊token [EOS]的位置。它被用来指示模型当前生成的句子已经结束

    CUTOFF_LEN = 256
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < CUTOFF_LEN
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt




@time_shower
def main():

     # 0---超参数
    # Setting for A100 - For RTX 3090
    #超参数
    MICRO_BATCH_SIZE = 4  # change to 4 for 3090
    BATCH_SIZE = 60  #为了加快训练，将默认的128加大为256，容易爆显存
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    #GRADIENT_ACCUMULATION_STEPS = 2
    EPOCHS = 20  # paper uses 3
    LEARNING_RATE = 3e-4  # from the original paper
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data 截断长度
    LORA_R = 8  #LORA中最重要的一个超参数，用于降维，LORA的秩
    LORA_ALPHA = 64 #alpha其实是个缩放参数，本质和learning rate相同，所以为了简化默认让alpha=rank
    LORA_DROPOUT = 0.05
    TRAIN_STEPS = 300


    # 1---分词器处理
    #设置基础模型与训练数据的路径
    TRAIN_DIR = "./train_dataset/IEMOCAP_train.json"
    output_dir = "lama2_model"
    # BASE_MODEL = "decapoda-research/llama-7b-hf"
    BASE_MODEL = "meta-llama/Llama-2-7b-hf"

    # 加载LLAMA的分词器, from_pretrained:加载预训练模型
    # tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL, add_eos_token=True)
    # #指定我们使用 [SEP] 标记来进行补充填充,而不是默认的 [PAD] 标记
    # tokenizer.pad_token = tokenizer.eos_token
    # #特殊token [EOS]的位置。它被用来指示模型当前生成的句子已经结束
    # tokenizer.pad_token_id = 0
    # tokenizer.padding_side = "left"


    # 2---开始进行数据加载与处理，可以替换为自己的数据集，前提是规范格式
    data = load_dataset("json", data_files=TRAIN_DIR)
    #打乱数据，设置训练集,截断、最大长度是256，可以包容96的数据，也可以更大设置为512，但是会慢
    train_val = data["train"].train_test_split(
        test_size=500, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].map(generate_and_tokenize_prompt)
    )
    print("-----------------------------------------")
    print("The dataset has been loaded.......Next load model and train model...")
    print("-----------------------------------------")


    # 3---开始模型与lora合并
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    # LlamaForCausalLM，可用于文本生成;  LlamaForSequenceClassification，可用于文本分类
    #模型加载自huggingface，load_in_8bit=True将模型权重和激活值量化为8位整数,从而减少内存和计算开销；device_map模型多卡并行
    model = LlamaForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, device_map=device_map,)
    #prepare_model_for_int8_training 对在Lora微调中使用LLM.int8()进行了适配
    model = prepare_model_for_int8_training(model)

    #设置一下LORA的config，调用一下get_peft_model方法，就获得了在原模型基础上的PEFT模型
    config = LoraConfig(r=LORA_R,
                        lora_alpha=LORA_ALPHA,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=LORA_DROPOUT,
                        bias="none",
                        task_type="CAUSAL_LM",)
    model = get_peft_model(model, config) #获得了在原模型基础上的PEFT模型
    print("peft...")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params


    # 4---多GPU模型训练


    print("now set the multi-GPU here...") #并行8个卡

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        # train_dataset=data["train"], #已经建立好的训练集，如果没有需要自己提前分好训练集和测试集
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            #max_steps=TRAIN_STEPS,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=20,
            save_steps=20,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            ddp_find_unused_parameters=False if ddp else None,
            fp16=True,
            logging_steps=1,
            load_best_model_at_end = True,
            output_dir=output_dir, #模型训练完的输出路径
            save_total_limit=3,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
             tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
         )
        #DataCollatorForLanguageModeling 实现了一个对文本数据进行随机 mask 的data collator
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False), #构建语言模型或者说是进行MLM任务时需要使用的数据收集器，该数据收集器会以一定概率（由参数mlm_probability控制）将序列中的Token替换成Mask标签

    )
    model.config.use_cache = False
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # model = torch.compile(model)
    print("Now training...")
    trainer.train(resume_from_checkpoint=False)	#resume_from_checkpoint 是否断点续训，开始训练
    print("....Model Has Been Trained....., Now, Save The Model....")
    model.save_pretrained(output_dir) #保存:它只包含 2 个文件: adapter_config.json 和 adapter_model.bin是保存经过训练的增量 PEFT 权重


    # 4---未来可以考虑将自己的模型上传
    #后期可以将自己训练好的模型上传到huggingface上，步骤是：
    # from huggingface_hub import notebook_login

    # notebook_login()

    # model.push_to_hub("affectLLM/Sentiment", use_auth_token=True) #这是我的账号




if __name__ == '__main__':
    main()
