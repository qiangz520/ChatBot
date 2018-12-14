#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : predata.py
# @Author    : ZJianbo
# @Date	     : 2018/10/13
# @Function  : 数据集的解析和前期准备

from helpers import *
from torch import optim
from model import *

"""
创建名为allDataWords的WordSeq类，用于统计单词。
    获取指定目录下的json文件，并进行解析将单词添加到allDataWords中。
    提取数据集中 [当前和下一句] 的 [单词长度和表情帧数] 均小于MAX_LENGTH的部分

allData: 所有json文件内容组
******************************
Creat:@ZJianbo @2018.10.13
Update:
"""
facevalue = FaceMaxMin()
allDataWords = WordSeq()
filenames = get_filename(DIR_PATH, "json")
print("Reading Words! Please wait...")
allData = []
# words_flag = os.path.exists(WORDS_PATH)
# faces_flag = os.path.exists(FACS_CPOINTS)
words_flag = False
faces_flag = False
for filename in filenames:
    text = loadfile_json(filename)
    for temp in text:
        if len(temp['text'].split(' ')) < MAX_LENGTH and \
                len(temp['text_next'].split(' ')) < MAX_LENGTH and \
                temp['facs_next_exist'] == 1 and temp['facs_prev_exist'] == 1 and \
                len(temp['facs']) < MAX_LENGTH and \
                len(temp['facs_next']) < MAX_LENGTH:
            allData.append(temp)
        if not words_flag:
            allDataWords.add_sentence(temp['text'])
            allDataWords.add_sentence(temp['text_next'])
            [allDataWords.add_sentence(hst) for hst in temp['text_history']]

# print(allDataWords.index2word)
allDataWords.load_words() if words_flag else allDataWords.save_words()
print("    no repeat words= ", allDataWords.n_words)
print("    sum bitches= ", len(allData))

print("Reading Faces! Please wait...")
allDataFaces = FacesCluster(allData, n_type=FACE_TYPE)
allDataFaces.load_faces() if faces_flag else (allDataFaces.run_cluster() and allDataFaces.save_faces())
print("    face types= ", allDataFaces.n_type)

# trainData = allData[0:1000]
# testData = allData[1000:1100]
trainData, testData, valData = split_data(allData)  # 4:1:1

trainDataset = TextDataset(allDataWords, allDataFaces, trainData)
trainDataloader = DataLoader(trainDataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

"""加载模型"""
Encoder_text_glo = EncoderTextBi(allDataWords.n_words, embeddingSize, hiddenSize).to(device)
Decoder_text_glo = DecoderText(hiddenSize, embeddingSize, allDataWords.n_words).to(device)
# Decoder_text_glo = AttnDecoderText(hiddenSize, allDataWords.n_words).to(device)
Encoder_face_glo = EncoderFace(allDataFaces.n_type, embeddingSize, hiddenSize).to(device)
Decoder_face_glo = DecoderFace(hiddenSize, embeddingSize, allDataFaces.n_type).to(device)
Encoder_HST_glo = EncoderHST(hiddenSize * 2, embeddingSize, hiddenSize).to(device)
EnOptimizer_text_glo = optim.SGD(Encoder_text_glo.parameters(), lr=LR_text)
DeOptimizer_text_glo = optim.SGD(Decoder_text_glo.parameters(), lr=LR_text)
EnOptimizer_face_glo = optim.Adam(Encoder_face_glo.parameters(), lr=LR_face)
DeOptimizer_face_glo = optim.Adam(Decoder_face_glo.parameters(), lr=LR_face)
Criterion_text_glo = nn.NLLLoss()
Criterion_face_glo = nn.NLLLoss()

if IsLoadModel:
    Encoder_text_glo.load_state_dict(torch.load('entext.pkl'))
    Decoder_text_glo.load_state_dict(torch.load('detext.pkl'))
    Encoder_face_glo.load_state_dict(torch.load('enface.pkl'))
    Decoder_face_glo.load_state_dict(torch.load('deface.pkl'))








