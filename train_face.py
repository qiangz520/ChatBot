#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : train.py
# @Author    : ZJianbo
# @Date	     : 2018/11/27
# @Function  : 开始训练

import random

from predata import *


"""训练单条数据。
train(tensor_text, tensor_face, batchsize)

Args:
    tensor_*: 包含了text/face 的内容和长度
    use_teacher_forcing: 是否使用“教师强迫”，即在decoder时，
                         以一定概率将target，而不是预测的输出，作为输入

******************************
Creat:@ZJianbo @2018.11.28
"""
def train(tensor_text, tensor_face, batchsize):
    """
    tensor_face[0][1][2]:
        Dim1: [0]input [1]target [2]prev
        Dim2: [0]content [1]size
        Dim3: 0~batchsize中的第几个
    """

    EnOptimizer_face_glo.zero_grad()
    DeOptimizer_face_glo.zero_grad()

    loss_size = 0
    loss = 0
    for num_batch in range(batchsize):
        loss_size += tensor_face[1][1][num_batch].numpy()
        #   隐藏层初始化
        enface_hidden = Encoder_face_glo.init_hidden()

        #   编码得到最后的隐藏层
        for ei in range(tensor_face[0][1][num_batch]):
            enface_output, enface_hidden = Encoder_face_glo(tensor_face[0][0][num_batch][ei], enface_hidden)

        deface_input = torch.tensor([[0]], device=device)
        deface_hidden = enface_hidden[:Decoder_face_glo.n_layers]

        teacher_forcing_ratio = 0.5
        # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        use_teacher_forcing = False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(tensor_face[1][1][num_batch]):
                decoder_output, decoder_hidden = Decoder_face_glo(deface_input, deface_hidden)
                loss += Criterion_face_glo(decoder_output, tensor_face[1][0][num_batch][di])
                deface_input = tensor_face[1][0][num_batch][di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(tensor_face[1][1][num_batch]):
                deface_output, deface_hidden = Decoder_face_glo(deface_input, deface_hidden)
                topv, topi = deface_output.topk(1)
                deface_input = topi.squeeze().detach()  # detach from history as input

                loss += Criterion_face_glo(deface_output, tensor_face[1][0][num_batch][di])
                if deface_input.item() == 1:
                    break

    loss.backward()

    EnOptimizer_face_glo.step()
    DeOptimizer_face_glo.step()

    return loss.item()/loss_size


"""训练数据集。
train_iters(train_dataloader, n_iters, print_every, plot_every)

Args:
    train_dataloader: 训练样本的dataloader
    n_iters: 训练次数
    print_every: 打印相关信息的间隔
    plot_every: 展示loss变化的间隔

******************************
Creat:@ZJianbo @2018.11.28
Update:
"""
def train_iters(train_dataloader, n_iters=10, print_every=100, plot_every=100):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for i_iter in range(1, n_iters + 1):
        for num, training_data in enumerate(train_dataloader):
            #   一个batch的数据
            tensor_text = training_data[0]
            tensor_face = training_data[1]

            loss = train(tensor_text, tensor_face, train_dataloader.batch_size)
            print_loss_total += loss
            plot_loss_total += loss

        if i_iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (CalculateTime().calc_time(start, i_iter / n_iters),
                                         i_iter, i_iter / n_iters * 100, print_loss_avg))
            evaluate_randomly(Encoder_face_glo, Decoder_face_glo, trainData, n=1)


"""单个表情组进行测试
@ZJianbo @2018.11.28
"""
def evaluate_face(encoder, decoder, face):

    input_tensor, input_size = Batch2Tensor().tensor_from_faces(allDataFaces, face)
    enface_hidden = Encoder_face_glo.init_hidden()

    #   编码得到最后的隐藏层
    for ei in range(input_size):
        enface_output, enface_hidden = encoder(input_tensor[ei], enface_hidden)
        # encoder_output, encoder_hidden = Encoder_text_glo(input_tensor[ei], encoder_hidden)
    deface_input = torch.tensor([[0]], device=device)
    deface_hidden = enface_hidden[:Decoder_face_glo.n_layers]
    decoded_faces = []
    for di in range(MAX_LENGTH):
        deface_output, deface_hidden = decoder(deface_input, deface_hidden)
        topv, topi = deface_output.data.topk(1)
        if topi.item() == 1:
            decoded_faces.append('<EOS>')
            break
        else:
            decoded_faces.append(topi.item())
        deface_input = topi.squeeze().detach()

    return decoded_faces


"""数据集中的表情组随机进行测试，可选择数目n
@ZJianbo @2018.11.28
"""
def evaluate_randomly(encoder, decoder, batches, n=3):
    for i in range(n):
        batch = random.choice(batches)
        out = evaluate_face(encoder, decoder, batch['facs'])
        print("out_size= ", len(out))
        print('>>>>> ', allDataFaces.get_faces_type(batch['facs']))
        print('===== ', allDataFaces.get_faces_type(batch['facs_next']))
        print('<<<<< ', out)


if __name__ == '__main__':
    train_iters(trainDataloader, n_iters=Train_Iters, print_every=Print_Every)
    if IsSaveModel:
        torch.save(Encoder_face_glo.state_dict(), 'enface.pkl')
        torch.save(Decoder_face_glo.state_dict(), 'deface.pkl')
    print("***** Face Evaluate *****")
    evaluate_randomly(Encoder_face_glo, Decoder_face_glo, testData, n=3)
