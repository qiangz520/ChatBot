#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : predata.py
# @Author    : ZJianbo
# @Date	     : 2018/10/13
# @Function  : 数据集的解析和前期准备

from helpers import *
from sklearn.model_selection import train_test_split
from torch import optim
from model import *

"""
创建名为allDataWords的WordSeq类，用于统计单词。
    获取指定目录下的json文件，并进行解析将单词添加到allDataWords中。
    提取数据集中 [当前和下一句] 的 [单词长度和表情帧数] 均小于MAX_LENGTH的部分
    
allData: 所有json文件内容组
******************************
Creat:@ZJianbo @2018.10.13
Update:
"""
allDataWords = WordSeq()
filenames = get_filename(DIR_PATH, "json")
print("Reading JsonFiles! Please wait...")
allData = []

for filename in filenames:
    text = loadfile_json(filename)
    for temp in text:
        if len(temp['text'].split(' ')) < MAX_LENGTH and \
                len(temp['text_next'].split(' ')) < MAX_LENGTH and \
                temp['facs_next_exist'] == 1 and len(temp['facs']) < MAX_LENGTH and \
                len(temp['facs_next']) < MAX_LENGTH:
            allData.append(temp)
        allDataWords.add_sentence(temp['text'])
        allDataWords.add_sentence(temp['text_next'])
        [allDataWords.add_sentence(hst) for hst in temp['text_history']]

# print(allDataWords.index2word)
print("no repeat words= %d ,sum words= %d" % (allDataWords.n_words, allDataWords.allwords))
print("sum bitches= ", len(allData))

# use sklearn (package) to split data (all data),random_state is fixed
trainData, testData, valData = split_data(allData)

# trainData = allData[0:1000]
# testData = allData[1000:1100]


# trainDataset = TextDataset(allDataWords, trainData)
trainDataset = TrainDataset(allDataWords, trainData)  # include text, history, face

trainDataloader = DataLoader(trainDataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
# print("data= ",training_data)
# print("old= ", trainData[0]['facs'][0])
# print("new= ", trainDataset[0][4][0])

"""加载模型"""
Encoder_text_glo = EncoderTextBi(allDataWords.n_words, embeddingSize, hiddenSize).to(device)
Decoder_text_glo = DecoderText(hiddenSize, embeddingSize, allDataWords.n_words).to(device)

Encoder_face_glo = EncoderFace(AU_size, hiddenSize).to(device)
Decoder_face_glo = DecoderFace(hiddenSize, AU_size).to(device)

Encoder_history_glo = EncoderTextBi(allDataWords.n_words, embeddingSize, hiddenSize).to(device)


EnOptimizer_text_glo = optim.Adam(Encoder_text_glo.parameters(), lr=LR_text)
DeOptimizer_text_glo = optim.Adam(Decoder_text_glo.parameters(), lr=LR_text)

EnOptimizer_face_glo = optim.Adam(Encoder_face_glo.parameters(), lr=LR_face)
DeOptimizer_face_glo = optim.Adam(Decoder_face_glo.parameters(), lr=LR_face)

EnOptimizer_history_glo = optim.Adam(Encoder_history_glo.parameters(),lr=LR_history)

Criterion_text_glo = nn.NLLLoss()
Criterion_face_glo = nn.MSELoss()
Criterion_history_glo = nn.MSELoss()
# Encoder_text_glo.load_state_dict(torch.load('entext.pkl'))
# Decoder_text_glo.load_state_dict(torch.load('detext.pkl'))
# Encoder_face_glo.load_state_dict(torch.load('enface.pkl'))
# Decoder_face_glo.load_state_dict(torch.load('deface.pkl'))
# Encoder_text_glo.load_state_dict(torch.load('encoder1028.pkl'))
# Decoder_text_glo.load_state_dict(torch.load('decoder1028.pkl'))







