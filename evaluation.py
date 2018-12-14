#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/11/22 21:53
# @Author : Qiangz
# @File : evaluation.py

from predata import allData
from helpers import split_data
from sklearn.model_selection import train_test_split
from model import *


# split data set to for mind reading test;train:test:val = 4:1:1
trainData, testData, valData = split_data(allData)