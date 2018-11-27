#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : parameters.py
# @Author    : ZJianbo
# @Date	     : 2018/10/28
# @Function  : 设置各个变量值

Train_Iters = 1000    # 训练次数,设置为0则不进行训练，只测试
Print_Every = 100     # 每隔多少打印一次数据
TestSentence = ["hello", "hi"]     # 测试语句,可添加修改
IsLoadModel = False
IsSaveModel = True

DIR_PATH = "/home/public/MoviechatData"     # 文件加载目录 MoviechatData MovieTest
WORDS_PATH = "index2word.json"      # 保存的单词
FACS_TYPE_PATH = "faceTypes.json"   # 保存的脸部类别
FACS_CPOINTS = "cPoints.json"   # 保存的聚类中心点

MAX_LENGTH = 15     # 句子单词和脸部帧数的最大长度
BATCH_SIZE = 16     # 批训练个数
FACE_TYPE = 200     # 脸部数据聚类个数

hiddenSize = 512   # 隐藏层
embeddingSize = 256     # 词嵌入维度
LR_text = 0.01   # 文本学习速率
LR_face = 0.01   # 表情学习速率


"""常量"""
AU_size = 47    # 脸部action unit个数
