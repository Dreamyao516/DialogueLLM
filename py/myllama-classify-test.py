
# 现在开始模型测试与分类！！！
# 4. 模型测试与分类！！

from datetime import date, datetime
import imp
import time
import traceback
import json
from datasets import load_dataset
from transformers import LlamaTokenizer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, LlamaForSequenceClassification
from sklearn import metrics
from transformers import  AutoModelForSequenceClassification
import pandas as pd
from pandas import DataFrame
import numpy as np

def time_shower(main):
    def call_main():
        print("\033[1;34m.....Main Start.....\033[0m")
        start = time.time()
        main()
        end = time.time()
        print("\n\033[1;34m.....Main End.....")
        print(f"-----共耗费了：{(end-start):.4f} 秒-----\033[0m")
    return call_main




#本身是不需要这个函数，但是因为前期 数据集处理时候，将label转化成了字符串，所以这个时候分类必须将"negative"重新转数字
def sentiment_score_to_name(score: str):
    if score == "Positive":
        return 1
    elif score == "Negative":
        return 0


def eval_performance(y_true, y_pred):
    # Precision
    print("Precision:\n\t", metrics.precision_score(y_true, y_pred, average='weighted'))

    # Recall
    print("Recall:\n\t", metrics.recall_score(y_true, y_pred, average='weighted'))

    # Accuracy
    print("Accuracy:\n\t", metrics.accuracy_score(y_true, y_pred))

    print("----------F1, Micro-F1, Macro-F1, Weighted-F1..----------------")
    print("----------**********************************----------------")

    # F1 Score
    print("F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='weighted'))

    # Micro-F1 Score
    print("Micro-F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='micro'))

    # Macro-F1 Score
    print("Macro-F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='macro'))

    # Weighted-F1 Score
    print("Weighted-F1 Score:\n\t", metrics.f1_score(y_true, y_pred, average='weighted'))

    print("------------------**********************************-------------------------")
    print("-------------------**********************************-------------------------")

    # ROC AUC Score
    # print("ROC AUC:\n\t", metrics.roc_auc_score(y_true, y_pred))

    # Confusion matrix
    print("Confusion Matrix:\n\t", metrics.confusion_matrix(y_true, y_pred))


@time_shower
def main():
    try:

        # 0----首先设置测试集路径 和 任务
        TEST_SET = "./test_dataset/meld_test_classify.json"
        TASK = "emotion"

        BASE_MODEL = "decapoda-research/llama-7b-hf"
        # BASE_MODEL = "meta-llama/Llama-2-7b-hf"
        LORA_MODEL = "lora-alpaca-wmy-classify-7B"
        # 1---分词器处理

        # 加载LLAMA的分词器, from_pretrained:加载预训练模型
        tokenizer = LlamaTokenizer.from_pretrained(
            BASE_MODEL)
        # 模型加载自huggingface，load_in_8bit=True将模型权重和激活值量化为8位整数,从而减少内存和计算开销；device_map模型多卡并行
        model = LlamaForSequenceClassification.from_pretrained(
                BASE_MODEL, load_in_8bit=True, device_map="auto",)

        # 读取保存peft模型及相关配置，使用PeftModel.from_pretrained(model, peft_model_id)方法加载模型
        model = PeftModel.from_pretrained(
                model, LORA_MODEL)


        #print(model)

        # 2---整个测试集测试
        # 如果是多个测试样本，可以参考如下：
        y_true = []
        y_pred = []

        with open(TEST_SET, "r", encoding="utf-8") as f:
            test_set = json.load(f)
            for i in range(len(test_set)):
                sample = test_set[i]
                if i>50:
                    break

               # 1 当处理分类任务时候，使用model = LlamaForSequenceClassification.from_pretrained。 尝试直接分类而不是生成。
                model = model.eval()
                new_sample = sample["text"]
                new_sample = tokenizer(new_sample, return_tensors="pt",)
                with torch.no_grad():
                    eval_logits = model(**new_sample).logits
                    predicted_class_id = eval_logits.argmax().item()
                    #eval_predictions = torch.argmax(eval_logits, dim=1)

                    print(predicted_class_id)
                    #print(eval_predictions)
                    #计算各项性能；Now we are going to compute the acc, weighted-f1, macro-f1, micro-f1, precision, recall
                    #这里走了弯路，如果直接用分类模型，那么前期不需要将标签数字化。
                    #开始处理标签
                    new_label = sample["label"]
                    #true_label = sentiment_score_to_name(new_label)

                    y_true.append(new_label)
                    y_pred.append(predicted_class_id)

        print(y_pred)
        print(y_true)

        # 将标签存入csv
        true_array = np.array(y_true)[:, np.newaxis]
        pre_array = np.array(y_pred)[:, np.newaxis]
        concatenate_array = np.concatenate((true_array, pre_array), axis=1)
        # print("concatenate_array", concatenate_array)
        # print("concatenate_array", concatenate_array.shape)
        data = DataFrame(concatenate_array, columns=["true_label", "pre_label"])
        data.to_csv("meld_result_2_7b_7ep.csv")

        # 计算一系列指标
        eval_performance(y_true, y_pred)


    except Exception as e:
        print(
            "\033[1;31m-------Error happened, Now we only show the running time------")
        traceback.print_exc()
        time_shower(main)


if __name__ == '__main__':
    main()
