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
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.cluster import KMeans

from torch.utils.data import *
from parameters import *
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True, precision=6)  # precision 是可选项
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

    def as_minutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def calc_time(self, since, percent):
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


def get_filename(filepath, filetype):
    filename = []
    ftype = '.' + filetype

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


"""保存json文件
Creat:@ZJianbo @2018.11.27
"""


def savefile_json(filename, data):
    with open(filename, 'w') as f:
        print(json.dumps(data), file=f)


"""将每个单词进行编号。
class WordSeq(object)

add_sentence(sentence)
    调用add_word(word)将输入的句子按单词进行添加、编号。
save_words()
    保存单词成json格式
load_words()
    加载json格式的单词

Args:
    allwords: 输入的所有的单词个数
    n_words： 去重后的单词个数
******************************
Creat:@ZJianbo @2018.10.13
Update:@ZJianbo @2018.11.27 增加save_words()和load_words()
"""


class WordSeq(object):
    def __init__(self):
        # self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
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

    def save_words(self):
        with open(WORDS_PATH, 'w') as f:
            print(json.dumps(self.index2word), file=f)

    def load_words(self):
        words = loadfile_json(WORDS_PATH)
        for i in range(2, len(words)):
            self.index2word[i] = words[str(i)]
            self.word2index[words[str(i)]] = i
            self.n_words += 1


"""
User kmeans algorithm to obtain gesture templates
Initialize kmeans

all_faces:List;Include all faces
kmeans:the obtained estimator by using sklearn.KMeans
getFacesIndexs:predict certain faces's cluster index, return a list

Create:@Qiangz @2018.11.26
Update:@ZJianbo @2018.11.27 修改__init__(),增加run_cluster()函数
                            增加_get_face()和_get_point()函数，用来获取表情和中心点
                            增加save_face()函数，用来保存聚类中心点和各表情类别
                            增加load_face()函数，用来加载聚类中心点和各表情类别
"""


class FacesCluster(object):
    def __init__(self, data, n_type):
        self.data = data
        self.n_type = n_type + 2
        self.c_points = {"0": np.zeros(AU_size).tolist(), "1": np.ones(AU_size).tolist()}
        self.face_types = {}
        self.kmeans = KMeans(n_clusters=n_type, max_iter=300, random_state=0, n_jobs=-1)

    def run_cluster(self):
        all_faces = []
        for temp in self.data:
            if temp["facs"]:
                for temp_face in temp["facs"]:
                    all_faces.append(temp_face)
            if temp["facs_prev"]:
                for temp_face in temp["facs_prev"]:
                    all_faces.append(temp_face)
        all_faces = np.array(all_faces)
        self.kmeans = self.kmeans.fit(all_faces)
        self._get_point(self.get_cluster_centers())
        all_points = []
        for i in range(self.n_type):
            all_points.append(self.c_points[str(i)])
        all_points = np.array(all_points)
        self.kmeans.cluster_centers_ = all_points

    def get_faces_type(self, faces):
        return self.kmeans.predict(np.array(faces).reshape(-1, AU_size))

    def get_cluster_centers(self):
        return self.kmeans.cluster_centers_

    def _get_face(self, facs):
        for i, temp in enumerate(facs):
            self.face_types[str(i)] = self.get_faces_type(temp["facs"]).tolist()

    def _get_point(self, points):
        for i, point in enumerate(points):
            self.c_points[str(i + 2)] = point.tolist()

    def save_faces(self):
        # self._get_point(self.get_cluster_centers())
        savefile_json(FACS_CPOINTS, self.c_points)
        # self._get_face(self.data)
        # savefile_json(FACS_TYPE_PATH, self.face_types)

    def load_faces(self):
        # self.face_types = loadfile_json(FACS_TYPE_PATH)
        self.c_points = loadfile_json(FACS_CPOINTS)
        all_points = []
        for i in range(self.n_type):
            all_points.append(self.c_points[str(i)])
        all_points = np.array(all_points)
        # self.kmeans = KMeans(init=(self.n_type, all_points), max_iter=10000, random_state=0, n_jobs=-1)
        self.kmeans = self.kmeans.fit(all_points)
        self.kmeans.cluster_centers_ = all_points


"""将每组数据转换成tensor类型。
class Batch2Tensor(object)

DataLoader里batchsize要求数据长度一致，故用句子用EOS_token填充，表情用AU_size维1矩阵填充。
Args:
    wordseq: WordSeq类，包含了数据集的所有单词
    batch: 输入的一组数据
    MAX_LENGTH: 每个句子最大长度
    input*/target* : 表示输入/输出
    tensor*/size* : 表示tensor类型的内容/长度
    text*/face* : 表示句子/表情
******************************
Creat:@ZJianbo @2018.10.13
Update: @ZJianbo @2018.10.14 将数据长度填充至MAX_LENGTH。

"""
SOS_token = 0
EOS_token = 1


class Batch2Tensor(object):
    def indexes_from_sentence(self, wordseq, sentence):
        return [wordseq.word2index[word] for word in sentence.split(' ')]

    def tensor_from_sentence(self, wordseq, sentence):
        indexes = self.indexes_from_sentence(wordseq, sentence)
        len_sentence = len(indexes)
        if len_sentence < MAX_LENGTH:
            for i in range(MAX_LENGTH - len_sentence):
                indexes.append(EOS_token)
            len_sentence += 1
        else:
            indexes = indexes[0:MAX_LENGTH]
            len_sentence = MAX_LENGTH
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1), len_sentence

    def tensor_from_faces(self, facecluster, face):
        face_temp = facecluster.get_faces_type(face).tolist()
        len_face = len(face_temp)
        if len_face < MAX_LENGTH:
            for i in range(MAX_LENGTH - len_face):
                face_temp.append(EOS_token)
            len_face += 1
        else:
            face_temp = face_temp[0:MAX_LENGTH]
            len_face = MAX_LENGTH
        return torch.tensor(np.array(face_temp), dtype=torch.long, device=device).view(-1, 1), len_face

    def tensors_from_batch(self, wordseq, facecluster, batch):
        # 12是历史信息个数10加上当前输入输出2
        # 0当前输入,1目标输出,2prev
        tensor_text = []
        tensor_face = []

        input_tensor_text, input_size_text = self.tensor_from_sentence(wordseq, batch['text'])
        tensor_text.append([input_tensor_text, input_size_text])
        target_tensor_text, target_size_text = self.tensor_from_sentence(wordseq, batch['text_next'])
        tensor_text.append([target_tensor_text, target_size_text])
        for i in range(10):
            input_tensor_HSTtext, input_size_HSTtext = \
                self.tensor_from_sentence(wordseq, batch['text_history'][i])
            tensor_text.append([input_tensor_HSTtext, input_size_HSTtext])

        input_tensor_face, input_size_face = self.tensor_from_faces(facecluster, batch['facs'])
        tensor_face.append([input_tensor_face, input_size_face])
        target_tensor_face, target_size_face = self.tensor_from_faces(facecluster, batch['facs_next'])
        tensor_face.append([target_tensor_face, target_size_face])
        input_tensor_facep, input_size_facep = self.tensor_from_faces(facecluster, batch['facs_prev'])
        tensor_face.append([input_tensor_facep, input_size_facep])

        return tensor_text, tensor_face


"""继承Dataset类，并重写方法。
class TextDataset(Dataset)

******************************
Creat:@ZJianbo @2018.10.13
Update:
"""


class TextDataset(Dataset):
    def __init__(self, data_words, data_faces, batches):
        self.dataWords, self.dataFaces, self.batches = data_words, data_faces, batches

    def __getitem__(self, index):
        return Batch2Tensor().tensors_from_batch(self.dataWords, self.dataFaces, self.batches[index])

    def __len__(self):
        return len(self.batches)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# plt.switch_backend('agg')

"""找到face每个AU的最大最小值"""


class FaceMaxMin(object):
    def __init__(self):
        self.max = np.zeros(AU_size)
        self.min = np.zeros(AU_size)

    def ismax(self, nowmax, value):
        if value > nowmax:
            return value
        else:
            return nowmax

    def ismin(self, nowmin, value):
        if value < nowmin:
            return value
        else:
            return nowmin

    def maxmin(self, values):
        for i in range(AU_size):
            self.max[i] = self.ismax(self.max[i], values[i])
            self.min[i] = self.ismin(self.min[i], values[i])


""" 
split data set to for mind reading test;train:test:val = 4:1:1
Create:@Qiangz @2018.11.28
"""


def split_data(dataset):
    temp, test = train_test_split(dataset, test_size=0.1, train_size=0.5, random_state=1000)
    # fix the random_state to keep the split in all experiments,split all data to temp:test = 5:1
    train, val = train_test_split(dataset, test_size=0.1, train_size=0.4, random_state=1000)
    # split the temp to train:val=4:1
    return train, test, val