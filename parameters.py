#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : parameters.py
# @Author    : ZJianbo
# @Date	     : 2018/10/28
# @Function  : 设置各个变量值

Train_Iters = 0    # 训练次数,设置为0则不进行训练，只测试
Print_Every = 1     # 每隔多少打印一次数据
TestSentence = ["thank you !", "hello"]     # 测试语句,可添加修改

DIR_PATH = "/home/public/MoviechatData"     # 文件加载目录
MAX_LENGTH = 15     # 句子单词和脸部帧数的最大长度
BATCH_SIZE = 16     # 批训练个数

hiddenSize = 512   # 隐藏层
embeddingSize = 256     # 词嵌入维度
LR_text = 0.01   # 文本学习速率
LR_face = 0.0005   # 表情学习速率


"""常量"""
AU_size = 47    # 脸部action unit个数
