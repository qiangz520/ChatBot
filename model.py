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
        return torch.zeros(1, 1, self.hidden_size, device=device)


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


"""History Encoder"""
class EncoderHST(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1):
        super(EncoderHST, self).__init__()
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


class EncoderFace(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1):
        super(EncoderFace, self).__init__()
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


class DecoderFace(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, n_layers=1):
        super(DecoderFace, self).__init__()
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


class AttnDecoderText(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, n_layers=1):
        super(AttnDecoderText, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size, self.output_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputd, hidden, encoder_outputs):
        embedded = self.embedding(inputd).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
