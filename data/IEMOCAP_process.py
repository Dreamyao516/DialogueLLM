# import os
# import glob
# import wave
# import python_speech_features as ps
# import numpy as np
# import pickle
# def read_wavefile(filename):
#     #开始读取wav文件
#     file = wave.open(filename,'r')
#     params = file.getparams() #获取得到的所有参数
#     n_channels, samp_with, fram_rate, wav_length = params[:4]
#     str_data = file.readframes(wav_length)
#     wave_data = np.fromstring(str_data,dtype=np.short)
#     time = np.arange(0,wav_length)*(1.0/fram_rate)
#     file.close()
#     return wave_data, time, fram_rate
# def read_IEMocap():
#     #设置数据库的路径
#     data_DIR = r"G:\IEMOCAP_full_release"
#     #设置训练数据和标签
#     train_data = []
#     train_label = []
#     #开始对数据库进行遍历过程
#     for speaker in os.listdir(data_DIR):
#         """
#         直接开始对每个进行搜索
#         并且设置缩小的范围 需要的数据进行查找
#         """
#         #因为还有其他的不是所需要的文件所以直接进行排除
#         if speaker[0] == 'S':
#             #语音存储的文件夹
#             speech_subdir = os.path.join(data_DIR,speaker,"sentences\\wav")
#             #语音标记的文件夹
#             speech_labledir = os.path.join(data_DIR,speaker,"dialog\\EmoEvaluation")
#             #将训练文件夹也保存
#             speech_file_dir = []
#             for sess  in os.listdir(speech_subdir):
#                 #sess 代表的是每个单独的文件夹 里面包含着每个单独的txt文件所以需要单独读取
#                 lable_text = speech_labledir+"\\"+sess+".txt"
#                 #获取到了 然后开始读取，这时要知道 读取文件需要用个list 或者是字典来进行存取
#                 emotion_lable = {}
#
#                 with open(lable_text,'r') as txt_read:
#                     """
#                     这里表达的是，文件读取第一行 看第一行如果有文件则进行保存对应的标签和结果 其中包含标注信息
#                     直到文件最后读取结束
#                     """
#                     while True:
#                         line = txt_read.readline()
#                         if not line:
#                             break
#                         if (line[0] == '['):
#                             t = line.split()
#                             emotion_lable[t[3]] = t[4]
#                 #--------------------------------------------------------
#                 """
#                 读取所有的音频文件
#                 """
#                 wava_file = os.path.join(speech_subdir,sess,'*wav')
#                 files= glob.glob(wava_file)#glob 主要是将目标的所有 来进行返回一个list集合
#                 for filename in files:
#                     #开始读取speech文件内的信息了 文件标签 存储数据内容
#                     wavaname = filename.split("\\")[-1][:-4] #得到文件名
#                     emotion = emotion_lable[wavaname] #通过对应来得到情感的对应标记
#                     #这里开始筛选是不是你需要的文件类型 比如你只想要hap ang neu sad 不要fear 那就可以不用把这个fear放入
#                     if emotion in ['hap','ang','neu','sad']:
#                         data, time, rate = read_wavefile(filename)
#                         mel_spec = ps.logfbank(data, rate, nfilt = 40)#滤波器的个数为40个
#                         time = mel_spec.shape[0]
#                         print("开始对{}文件的计算".format(filename))
#                         #开始对不满足时间少于300的进行padding 0
#                         if time <=300:
#                             padding_data = mel_spec
#                             #后面补充0
#                             #padding_data=np.pad(padding_data,((0,300-padding_data.shape[0]),(0,0)),'constant',constant_value=0)
#                             padding_data = np.pad(padding_data, ((0, 300 - padding_data.shape[0]), (0, 0)), 'constant',
#                                           constant_values=0)
#
#                             train_data.append(padding_data)
#                             train_label.append(emotion)
#                             speech_file_dir.append(filename)
#                         else:
#                             begin = 0
#                             end = 300
#                             padding_data = mel_spec[begin:end,:]
#                             train_data.append(padding_data)
#                             train_label.append(emotion)
#                             speech_file_dir.append(filename)
#     #写入pkl文件中
#     print("开始写入文件中")
#     f = open('./IEMOCAP.pkl', 'wb')
#     #将数据 放入pickle中保存
#     pickle.dump((train_data,train_label,speech_file_dir),f)
#     f.close()
#     print("写入完毕")
#     pass
# def read_picklefile():
#     f = open('./IEMOCAP.pkl','rb')
#     train_data,train_lable,speech_file = pickle.load(f)
#     print(len(train_lable))
# if __name__ == "__main__":
#     read_picklefile()
import json
import os

import pandas as pd


def get_txt_files(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                filename = os.path.join(root, file)
                # print(get_label(filename))
                L.append(filename)
        return L


def get_IEMOCAP_filename(IEMOCAP_dir):
    all_data = []
    Sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    target_dir = "/dialog/EmoEvaluation/"
    wav_dir = "/dialog/transcriptions/"
    i = 0;j=0;
    for Session in Sessions:
        file_dir = IEMOCAP_dir + Session + target_dir
        sentence_dir = IEMOCAP_dir + Session + wav_dir
        # 读取该Session下的所有txt文件
        txt_files = get_txt_files(file_dir)
        sen_files = get_txt_files(sentence_dir)
        # 读取单个txt文件，获取情感标签,sen_file,sentence_dir

        for txt_file,sen_file in zip(txt_files,sen_files):
            # last_folder = (txt_file.split("/")[-1]).split(".")[0]
            # 转换到pandas，方便操作
            print("i:", i)
            i += 1
            data = pd.read_csv(txt_file, delimiter="\n", skiprows=1, names='a')
            data['a'] = data['a'].astype(str)
            # 取第一列，其中包含切片后的语音文件+情感分类标签
            filter_data = [x for x in data['a'] if '[' in x]
            print(filter_data)
            datas = pd.read_csv(sen_file, delimiter="\n", skiprows=1, names='a')
            datas['a'] = datas['a'].astype(str)
            # 取第一列，其中包含切片后的语音文件+情感分类标签
            sen_data = [x for x in datas['a'] if '[' in x]
            print(sen_data)
            for file in filter_data:
                j+=1
                values = file.split("\t")
                filename = values[1]
                print(filename)
                target = values[2]
                label = get_IEMOCAP_9target(target)
                print(target)
                for sen in sen_data:
                    sen_values1 = sen.split("]:")
                    sen_values = sen_values1[0].split(" [")
                    sen_name = sen_values[0]
                    print(sen_name)
                    # sen_values = sen.split(" [")
                    # sen_name = sen_values[0]
                    # sen_values1 = sen_values[1].split("]:")
                    # print(sen_values1)
                    sentence = sen_values1[1]
                    # if (target == None):
                    #     print(target)
                    #     continue
                    if (filename == sen_name):
                        if(label!="Other"):
                            result = {
                                "instruction": "Given the Speaker and Context, detect the emotion label of the input utterance from ['Neutral','Joy','Sadness','Anger', 'Fear', 'surprise', 'excitement', 'frustration','other'].",
                                "input": sentence, "output": label}
                            all_data.append(result)
                            break
        print(i)
        print("j:",j)
            #print(j)
            # 是否包含某字符串
            # print(txt_file, len(filter_data))
            #
            # j = 0;
            # for sen_file, file in zip(sen_files, filter_data):
            #     datas = pd.read_csv(sen_file, delimiter="\n", skiprows=1, names='a')
            #     datas['a'] = datas['a'].astype(str)
            #     # 取第一列，其中包含切片后的语音文件+情感分类标签
            #     sen_data = [x for x in datas['a'] if '[' in x]
            #     print(sen_data)
            #     print(j)
            #     j += 1;
            #     # 是否包含某字符串
            #     # print(txt_file, len(filter_data))
            #
            #     # for file in filter_data:
            #     values = file.split("\t")
            #     filename = values[1]
            #     target = values[2]
            #     label = get_IEMOCAP_9target(target)
            #     print(target)
            #     for sen in sen_data:
            #         sen_values1 = sen.split("]:")
            #         sen_values = sen_values1[0].split(" [")
            #         sen_name = sen_values[0]
            #         # sen_values = sen.split(" [")
            #         # sen_name = sen_values[0]
            #         # sen_values1 = sen_values[1].split("]:")
            #         # print(sen_values1)
            #         sentence = sen_values1[1]
            #         # if (target == None):
            #         #     print(target)
            #         #     continue
            #         if (filename == sen_name):
            #             result = {
            #                 "instruction": "Given the Speaker and Context, detect the emotion label of the input utterance from ['Neutral','Joy','Sadness','Anger', 'Fear', 'surprise', 'excitement', 'frustration','other'].",
            #                 "input": sentence, "output": label}
            #
            #             all_data.append(result)
            #             break

    return all_data

    #     i=0;
    #     for txt_file in txt_files:
    #         #last_folder = (txt_file.split("/")[-1]).split(".")[0]
    #         # 转换到pandas，方便操作
    #         print("i:",i)
    #         i+=1
    #
    #         data = pd.read_csv(txt_file, delimiter="\n", skiprows=1, names='a')
    #         data['a'] = data['a'].astype(str)
    #         # 取第一列，其中包含切片后的语音文件+情感分类标签
    #         filter_data = [x for x in data['a'] if '[' in x]
    #         print(filter_data)
    #         # 是否包含某字符串
    #         #print(txt_file, len(filter_data))
    #
    #         j=0;
    #         for sen_file,file in zip(sen_files,filter_data):
    #             datas = pd.read_csv(sen_file, delimiter="\n", skiprows=1, names='a')
    #             datas['a'] = datas['a'].astype(str)
    #             # 取第一列，其中包含切片后的语音文件+情感分类标签
    #             sen_data = [x for x in datas['a'] if '[' in x]
    #             print(sen_data)
    #             print(j)
    #             j+=1;
    #             # 是否包含某字符串
    #             #print(txt_file, len(filter_data))
    #
    #         # for file in filter_data:
    #             values = file.split("\t")
    #             filename = values[1]
    #             target = values[2]
    #             label = get_IEMOCAP_9target(target)
    #             print(target)
    #             for sen in sen_data:
    #                 sen_values1 = sen.split("]:")
    #                 sen_values = sen_values1[0].split(" [")
    #                 sen_name = sen_values[0]
    #                 # sen_values = sen.split(" [")
    #                 # sen_name = sen_values[0]
    #                 # sen_values1 = sen_values[1].split("]:")
    #                 #print(sen_values1)
    #                 sentence = sen_values1[1]
    #                 # if (target == None):
    #                 #     print(target)
    #                 #     continue
    #                 if(filename==sen_name):
    #                     result = {"instruction":"Given the Speaker and Context, detect the emotion label of the input utterance from ['Neutral','Joy','Sadness','Anger', 'Fear', 'surprise', 'excitement', 'frustration','other'].","input":sentence,"output":label}
    #
    #                     all_data.append(result)
    #                     break
    #
    # return all_data


def get_IEMOCAP_9target(target):
    # 'ang': 0, anger 愤怒
    # 'hap': 1, happiness 快乐，幸福
    # 'exc': 1, excitement 激动，兴奋
    # 'sad': 3, sadness 悲伤，悲痛
    # 'fru': 4, frustration 懊恼，沮丧
    # 'fea': 5, fear 害怕，畏惧
    # 'sur': 6, surprise 惊奇，惊讶
    # 'neu': 7, neutral state 中性
    # 'xxx': 8, other 其它

    if (target == "ang"):
        target = "Anger"
    elif (target == "hap"):
        target = "Joy"
    elif (target == "exc"):
        target = "Excitement"
    elif (target == "sad"):
        target = "Sadness"
    elif (target == "fru"):
        target = "Frustration"
    elif (target == "fea"):
        target = "Fear"
    elif (target == "sur"):
        target = "Surprise"
    elif (target == "neu"):
        target = "Neutral"
    elif (target == "xxx"):
        target = "Other"
    else:
        print(target)
        target = None
        print("关键词提取错误，经检查程序，已默认跳过该条数据")

    return target


def get_IEMOCAP_4target(target):
    # 'ang': 0, anger 愤怒
    # 'hap': 1, happiness 快乐，幸福
    # 'sad': 2, sadness 悲伤，悲痛
    # 'neu': 3, neutral state 中性
    #

    if (target == "ang"):
        target = 0
    elif (target == "hap"):
        target = 1
    elif (target == "sad"):
        target = 2
    elif (target == "neu"):
        target = 3
    else:
        print(target)
        target = None
        print("关键词提取错误，经检查程序，已默认跳过该条数据")

    return target

wav_path="D:/IEMOCAP/"
#spec_path="./features/ICMOCAP_Spec(4kinds)/"
filenames=get_IEMOCAP_filename(wav_path)
#print(filenames)
output_dir = "D:/IEMOCAP/IEMOCAP.json"
with open(output_dir, "w") as f:
    json.dump(filenames, f)

#将文件路径和对应的标签进行拆解
# filenames,targets=zip(*filenames)
# for i,filename in enumerate(filenames):
# 	label=targets[i]
