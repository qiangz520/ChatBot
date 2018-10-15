#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : helpers.py
# @Author    : ZJianbo
# @Date	     : 2018/10/13
# @Function  : 函数和类

import os
import json
import time
import math

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.utils.data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = {}".format(device))


"""计算已耗时和剩余时间。
class CalculateTime(object)

Args:
calc_time(since,percent)
    since: 开始时间
    percent: 已完成百分比
    
Returns:
    %s (- %s)
    已耗时 （- 剩余时间）
******************************
Creat:@ZJianbo @2018.10.13
Update:
"""
class CalculateTime(object):
    def __init__(self):
        pass

    def as_minutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def calc_time(self,since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (self.as_minutes(s), self.as_minutes(rs))


"""获取目录下的目标种类文件名(含路径)。
get_filename(filepath,filetype)

Args:
    filepath: 指定的目标路径
    filetype: 想要获得的目标文件种类其后缀名，如json,txt等。若为all,则为所有文件

Returns:
    filename: 文件名列表
******************************
Creat:@ZJianbo @2018.10.13
Update:
"""
def get_filename(filepath,filetype):
    filename = []
    ftype = '.'+filetype

    for root, dirs, files in os.walk(filepath):
        for file in files:
            if filetype == "all" or os.path.splitext(file)[1] == ftype:
                filename.append(os.path.join(root, file))

    return filename


"""加载目标json文件。
loadfile_json(filename)

Args:
    filename: 目标json文件(路径+文件名)

Returns:
    text: json文件解析后的列表
******************************
Creat:@ZJianbo @2018.10.13
Update:
"""
def loadfile_json(filename):
    with open(filename, "r", encoding='utf-8') as f:
        text = json.load(f)
    return text


"""将每个单词进行编号。
class WordSeq(object)

add_sentence(sentence)
调用add_word(word)将输入的句子按单词进行添加、编号。

Args:
    allwords: 输入的所有的单词个数
    n_words： 去重后的单词个数
******************************
Creat:@ZJianbo @2018.10.13
Update:
"""
class WordSeq(object):
    def __init__(self):
        #self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.allwords = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        self.allwords += 1
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


"""将每组数据转换成tensor类型。
class Batch2Tensor(object)

DataLoader里batchsize要求数据长度一致，故用EOS_token填充。
Args:
    wordseq: WordSeq类，包含了数据集的所有单词
    batch: 输入的一组数据
    MAX_LENTH: 每个句子最大长度
    input_len, output_len: 每个句子的真实长度(包含一个EOS)
******************************
Creat:@ZJianbo @2018.10.13
Update: @ZJianbo @2018.10.14 将数据长度填充至MAX_LENTH。

"""
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50
class Batch2Tensor(object):
    def indexes_from_sentence(self, wordseq, sentence):
        return [wordseq.word2index[word] for word in sentence.split(' ')]

    def tensor_from_sentence(self, wordseq, sentence):
        indexes = self.indexes_from_sentence(wordseq, sentence)
        len_sentence = len(indexes)
        for i in range(MAX_LENGTH - len_sentence):
            indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1), len_sentence+1

    def tensors_from_batch(self, wordseq, batch):
        input_tensor, input_len = self.tensor_from_sentence(wordseq, batch['text'])
        target_tensor, output_len = self.tensor_from_sentence(wordseq, batch['text_next'])
        return input_tensor, target_tensor, input_len, output_len

"""继承Dataset类，并重写方法。
class TextDataset(Dataset)

******************************
Creat:@ZJianbo @2018.10.13
Update:
"""
class TextDataset(Dataset):
    def __init__(self, data_words, batches):
        self.dataWords, self.batches = data_words, batches

    def __getitem__(self, index):
        return Batch2Tensor().tensors_from_batch(self.dataWords, self.batches[index])

    def __len__(self):
        return len(self.batches)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

plt.switch_backend('agg')
