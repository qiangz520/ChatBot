#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : model.py
# @Author    : ZJianbo
# @Date	     : 2018/10/13
# @Function  : 训练模型

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import *

"""单向GRU"""
class EncoderTextSi(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(EncoderTextSi, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input_data, hidden):
        embedded = self.embedding(input_data).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return None


"""双向GRU"""
class EncoderTextBi(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1):
        super(EncoderTextBi, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True)

    def forward(self, input_data, hidden):
        embedded = self.embedding(input_data).view(1, 1, -1)
        outputs, hidden = self.gru(embedded, hidden)

        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden

    def init_hidden(self):
        return None
        # return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderText(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, n_layers=1):
        super(DecoderText, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data, hidden):
        output = self.embedding(input_data).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return None
        # return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderFace(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderFace, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)

    def forward(self, input_data, hidden):
        outputs, hidden = self.gru(input_data.view(1,1,-1), hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

    def init_hidden(self):
        return None


class DecoderFace(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderFace, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.inl = nn.Linear(output_size, output_size)
        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
       # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data, hidden):
        output = F.relu(input_data.view(1, 1, -1))
        output = self.inl(output.float())
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0].float())
      #  output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return None
