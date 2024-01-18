# 现在开始模型测试与生成
# 4. 模型测试与生成

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
import pandas as pd
from pandas import DataFrame
import numpy as np
import  random

def time_shower(main):
    def call_main():
        print("\033[1;34m.....Main Start.....\033[0m")
        start = time.time()
        main()
        end = time.time()
        print("\n\033[1;34m.....Main End.....")
        print(f"-----共耗费了：{(end - start):.4f} 秒-----\033[0m")

    return call_main


def generate_test_prompt(data_point):
    # create prompts from the loaded dataset and tokenize them
    if data_point["input"]:
        if "Context" not in data_point:
            return f"""Please perform Classification task. Given the sentence, assign the correct label. Return label only without any other text.
            ### Instruction:
            {data_point["instruction"]}

            ### Input:
            {data_point["input"]}

            ### Response:
            """
        else:
            return f"""Please perform Classification task. Given the sentence, assign the correct label. Return label only without any other text.
            ### Instruction:
            {data_point["instruction"]}

            ### Context:
            {data_point["Context"]}

            ### Input:
            {data_point["input"]}

            ### Response:
            """

    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        {data_point["instruction"]}

        ### Response:
        {data_point["output"]}"""

# 本身是不需要这个函数，但是因为前期 数据集处理时候，将label转化成了字符串，所以这个时候分类必须将"negative"重新转数字
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


def post_process(task, response: str):
    response = response.strip().lower()
    response = response.split(".")[0].strip()
    response = response.split("\n")[0].strip()
    print("R_split:", response)
    labels_dict = {"sentiment": ["positive", "neutral", "negative", "pos", "neg", "neu", "0", "1"],
                   "emotion": ["happy", "happiness", "joy", "fear", "scare", "sad", "sadness", "disgust", "disgusting",
                               "hate", "surprise", "anger", "angry", "rage", "neutral", "amazed", "surprising",
                               "amazing", "0","excitement","frustration"],
                   "sarcasm": ["sarcastic", "sarcasm", "irony", "satire", "ironic", "non-sarcastic", "non-ironic"],
                   "humor": ["humor", "fun", "non-humor", "non-fun", "not fun", "not humor"],
                   "offensive": ["offensive", "offend", "non-offensive"],
                   "enthusiasm": ["monotonous", "normal", "enthusiastic"]}

    synonymy_dict = {"sentiment": {("positive", "pos", "1"): 1, ("neutral", "neu"): 0, ("negative", "neg", "0"): 2},
                     "emotion": {("happy", "happiness", "joy", "glad"): 1, ("neutral", "neu", "0"): 0,
                                 ("sad", "sadness"): 2, ("anger", "angry", "rage"): 3, ("fear", "scare"): 4,
                                 ("excitement"): 5,
                                 ("surprise", "amazing", "amazed", "surprising"): 6,
                                 ("frustration"): 7},
                     "sarcasm": {("sarcastic", "sarcasm", "irony", "satire", "ironic"): 1,
                                 ("non-sarcastic", "non-ironic"): 0},
                     "humor": {("humor", "fun", "non-humor"): 1, ("non-fun", "not fun", "not humor"): 0},
                     "offensive": {("offensive", "offend"): 1, ("non-offensive"): 0},
                     "enthusiasm": {("monotonous"): 0, ("normal"): 1, ("enthusiastic"): 2},
                     }

    selected_task = labels_dict.get(task)
    for i in range(len(selected_task)):
        category = selected_task[i]
        if response.find(category) != -1:
            for key, value in synonymy_dict[task].items():
                if category in key:
                    # print(category)
                    return value
                    break

            break


def generate_true_labels(task, TEST_SET):
    true_labels = []
    i = 0;
    with open(TEST_SET, "r", encoding="utf-8") as f:
        for test_set in json.load(f):
            # if i>9:
            #     break
            # i = i+1
            output = test_set["output"].lower()
            label = post_process(task, output)
            true_labels.append(int(label))

    return true_labels


@time_shower
def main():
    try:

        # 0----首先设置测试集路径 和 任务
        TEST_SET = "./test_dataset/IEMOCAP_test.json"
        TASK = "emotion"

        #BASE_MODEL = "decapoda-research/llama-7b-hf"
        BASE_MODEL = "meta-llama/Llama-2-7b-hf"
        LORA_MODEL = "lama2_IEMOCAPend_10"
        outputdir = "./generate_test/lama13_IEMOCAP.csv"
        pred_labels = []
        true_labels = generate_true_labels(TASK, TEST_SET)  # 先将测试集的 标签 数字化，方便后续直接评测计算。

        # 1---分词器处理

        # 加载LLAMA的分词器, from_pretrained:加载预训练模型
        tokenizer = LlamaTokenizer.from_pretrained(
            BASE_MODEL)
        # 模型加载自huggingface，load_in_8bit=True将模型权重和激活值量化为8位整数,从而减少内存和计算开销；device_map模型多卡并行
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL, load_in_8bit=True, device_map="auto", )

        #读取保存peft模型及相关配置，使用PeftModel.from_pretrained(model, peft_model_id)方法加载模型
        model = PeftModel.from_pretrained(
            model, LORA_MODEL, adapter_name="my_alpaca")

        # 1----[单个样例测试]
        # 设计了一个测试用的prompt，用于情感分析，试一试结果.
        # PROMPT = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.
        # ### Instruction:
        # Assign a correct sentiment label of the tweets from ['negative', 'neural', 'positive'].
        # Return label only.
        # ### Input:
        # I am a kind of paranoiac in reverse. I suspect people of plotting to make me happy.
        # ### Response:'''

        # 2---整个测试集测试
        # 如果是多个测试样本，可以参考如下：
        with open(TEST_SET, "r", encoding="utf-8") as f:
            test_set = json.load(f)
            for i in range(len(test_set)):
                # if i > 9:
                #     break
                sample = test_set[i]
                # print(sample)
                PROMPT = generate_test_prompt(sample)
                inputs = tokenizer(PROMPT, return_tensors="pt", )
                input_ids = inputs["input_ids"].cuda()  # 将输入张量移动到 GPU 上
                # top_p 已知生成各个词的总概率是1（即默认是1.0）如果top_p小于1，则从高到低累加直到top_p，取这前N个词作为候选; repetition_penalty 重复处罚的参数
                generation_config = GenerationConfig(
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    # temperature=0.1,
                    # top_p=0.75,
                    top_k=40,
                    num_beams=4,
                )
                # temperature=0.1, top_p=0.95, repetition_penalty=1.15,
                print("Now we are Generating...\n")

                # 3--- 当处理生成任务时候，使用model = LlamaForCausalLM.from_pretrained。 生成任务而不是分类---以后情感对话生成！
                generation_output = model.generate(input_ids=input_ids,
                                                   generation_config=generation_config,
                                                   return_dict_in_generate=True,
                                                   output_scores=True,
                                                   max_new_tokens=256, )

                s = generation_output.sequences[0]
                output = tokenizer.decode(s)
                # output = output[2:output.find('"')]
                print("output:", output)
                if "### Response:" in output:
                    response = output.split("### Response:")[1].strip()
                else:
                    print("@@@@@@@@@@@@@")
                    response = "Neutral"
                print("Response:", response)
                print("-----------------------------------------")
                label = post_process(TASK, response)
                random_list = [0, 1, 2, 3, 4, 5, 6]
                if (label == None):
                    label = random.choice(random_list)
                pred_labels.append(label)
                print(label)
                print("***")
                print(i)
                print("..............Now Post-processing Respnese and Generating Labels..............")

        # 将标签存入csv
        true_array = np.array(true_labels)[:, np.newaxis]
        pre_array = np.array(pred_labels)[:, np.newaxis]
        concatenate_array = np.concatenate((true_array, pre_array), axis=1)
        # print("concatenate_array", concatenate_array)
        # print("concatenate_array", concatenate_array.shape)
        data = DataFrame(concatenate_array, columns=["true_label", "pre_label"])
        data.to_csv(outputdir)

        # 4----开始模型评测计算
        print(".........Now Computing P, R, F1 Scores..........")

        eval_performance(true_labels, pred_labels)
    except Exception as e:
        print(
            "\033[1;31m-------Error happened, Now we only show the running time------")
        traceback.print_exc()
        time_shower(main)


if __name__ == '__main__':
    main()
