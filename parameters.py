#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : parameters.py
# @Author    : ZJianbo
# @Date	     : 2018/10/28
# @Function  : 设置各个变量值

Train_Iters = 200    # 训练次数,设置为0则不进行训练，只测试
Print_Every = 20     # 每隔多少打印一次数据
TestSentence = ["what is your name ?", "hello robot"]     # 测试语句,可添加修改
IsLoadModel = False
IsSaveModel = False

DIR_PATH = "/home/public/ChatBotData"     # 文件加载目录 ChatBotData ChatBotTest
WORDS_PATH = "./Files/index2word.json"      # 保存的单词
FACS_TYPE_PATH = "./Files/faceTypes.json"   # 保存的脸部类别
FACS_CPOINTS = "./Files/cPoints.json"   # 保存的聚类中心点

MAX_LENGTH = 15     # 句子单词和脸部帧数的最大长度
BATCH_SIZE = 16     # 批训练个数
FACE_TYPE = 300     # 脸部数据聚类个数

hiddenSize = 512   # 隐藏层
embeddingSize = 256     # 词嵌入维度
LR_text = 0.001   # 文本学习速率
LR_face = 0.001   # 表情学习速率
LR_hst = 0.001

"""常量"""
AU_size = 21    # 脸部action unit个数
