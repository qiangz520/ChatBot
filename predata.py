#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : predata.py
# @Author    : ZJianbo
# @Date	     : 2018/10/13
# @Function  : 数据集的解析和前期准备

import helpers as hp

from torch import optim
from model import *

DIR_PATH = "./data"

"""
创建名为allSentences的WordSeq类，用于统计单词。
获取指定目录下的json文件，并进行解析将单词添加到allSentences中。

allText: 所有json文件内容组
******************************
Creat:@ZJianbo @2018.10.13
Update:
"""
allDataWords = hp.WordSeq()
filenames = hp.get_filename(DIR_PATH, "json")
print("Reading JsonFiles! Please wait...")
allText = []
for filename in filenames:
    text = hp.loadfile_json(filename)
    for temp in text:
        allText.append(temp)
        allDataWords.add_sentence(temp['text'])
        allDataWords.add_sentence(temp['text_next'])
        [allDataWords.add_sentence(hst) for hst in temp['text_history']]

# print(allDataWords.index2word)
print("no repeat words= %d ,sum words= %d" % (allDataWords.n_words, allDataWords.allwords))
print("sum batches=", len(allText))

trainText = allText[0:1000]
trainDataset = hp.TextDataset(allDataWords, trainText)
trainDataloader = hp.DataLoader(trainDataset, shuffle=True, batch_size=100)


"""参数设置"""
HiddenSize_glo = 1024
EmbeddingSize_glo = 512
LearningRate_glo = 0.01
Encoder_glo = EncoderRNN(allDataWords.n_words, EmbeddingSize_glo, HiddenSize_glo).to(device)
Decoder_glo = DecoderRNN(HiddenSize_glo, EmbeddingSize_glo, allDataWords.n_words).to(device)
EncoderOptimizer_glo = optim.SGD(Encoder_glo.parameters(), lr=LearningRate_glo)
DecoderOptimizer_glo = optim.SGD(Decoder_glo.parameters(), lr=LearningRate_glo)
Criterion_glo = nn.NLLLoss()
# Encoder_glo.load_state_dict(torch.load('encoder.pkl'))
# Decoder_glo.load_state_dict(torch.load('decoder.pkl'))







