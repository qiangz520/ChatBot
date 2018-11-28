#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/11/22 21:53
# @Author : Qiangz
# @File : evaluation.py

from predata import allData
from sklearn.model_selection import train_test_split
from model import *


# split data set to for mind reading test;train:test:val = 4:1:1
def split_data(dataset):
    temp, test = train_test_split(dataset, test_size=0.1, train_size=0.5, random_state=1000)
    # fix the random_state to keep the split in all experiments,split all data to temp:test = 5:1
    train, val = train_test_split(dataset, test_size=0.1, train_size=0.4, random_state=1000)
    # split the temp to train:val=4:1
    return train, test, val


trainData, testData, valData = split_data(allData)

