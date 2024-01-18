# -*- coding: utf-8 -*-
# 现在开始模型测试与生成
# 4. 模型测试与生成

from datetime import date, datetime
import imp
import time
import traceback

from sklearn import metrics
import pandas as pd
from pandas import DataFrame
import numpy as np
import csv

def time_shower(main):
    def call_main():
        print("\033[1;34m.....Main Start.....\033[0m")
        start = time.time()
        main()
        end = time.time()
        print("\n\033[1;34m.....Main End.....")
        print(f"-----共耗费了：{(end - start):.4f} 秒-----\033[0m")

    return call_main


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
    #print("ROC AUC:\n\t", metrics.roc_auc_score(y_true, y_pred))

    # Confusion matrix
    print("Confusion Matrix:\n\t", metrics.confusion_matrix(y_true, y_pred))

def get_column_elements(file_path, column_index_output, column_index_answer):

    with open(file_path) as csvfile:

        # Return a reader object which will
        # iterate over lines in the given csvfile.
        readCSV = csv.reader(csvfile, delimiter=',')
        # for row in readCSV:
        #     print(row)
        #     print(row[0])
        #     print(row[0], row[1], row[2], )
        column_elements_output = []
        column_elements_answer = []
        for row_index in readCSV:  # 从第二行开始读取，避免读取表头
            if row_index[column_index_output] == "true_label":
                continue
            elif row_index[column_index_answer] == "pre_label":
                continue
            column_elements_output.append(row_index[column_index_output].lower())
            column_elements_answer.append(row_index[column_index_answer].lower())
        return column_elements_output,column_elements_answer



@time_shower
def main():
    try:

        file_path = "./generate_test/lama2.csv"

        y_true_index = 1
        y_pred_index = 2
        true_labels, pred_labels = get_column_elements(file_path, y_true_index, y_pred_index)



        eval_performance(true_labels, pred_labels)
    except Exception as e:
        print(
            "\033[1;31m-------Error happened, Now we only show the running time------")
        traceback.print_exc()
        time_shower(main)


if __name__ == '__main__':
    main()
