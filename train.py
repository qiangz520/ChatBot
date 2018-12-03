#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : train.py
# @Author    : ZJianbo
# @Date	     : 2018/11/29
# @Function  : 开始训练

import random

from predata import *

def train(tensor_text, tensor_face, batchsize):

    EnOptimizer_text_glo.zero_grad()
    DeOptimizer_text_glo.zero_grad()
    Encoder_HST_glo.zero_grad()
    # EnOptimizer_face_glo.zero_grad()
    # DeOptimizer_face_glo.zero_grad()

    loss_text_size = 0
    loss_text = 0

    for num_batch in range(batchsize):
        loss_text_size += tensor_text[1][1][num_batch].numpy()

        #   <HST>
        entextHST_hidden = []
        for i in range(10):
            entext_hidden = Encoder_text_glo.init_hidden()
            for ei in range(tensor_text[i+2][1][num_batch]):
                entext_output, entext_hidden = Encoder_text_glo(tensor_text[i+2][0][num_batch][ei], entext_hidden)
            entextHST_hidden.append(entext_hidden[:Decoder_text_glo.n_layers])

        history_hidden = Encoder_HST_glo.init_hidden()
        for i in range(10):
            history_output, history_hidden = Encoder_HST_glo(entextHST_hidden[9-i], history_hidden)
        #   </HST>

        entext_hidden = Encoder_text_glo.init_hidden()
        #   编码得到最后的隐藏层
        for ei in range(tensor_text[0][1][num_batch]):
            entext_output, entext_hidden = Encoder_text_glo(tensor_text[0][0][num_batch][ei], entext_hidden)

        detext_input = torch.tensor([[SOS_token]], device=device)
        detext_hidden = entext_hidden[:Decoder_text_glo.n_layers] + history_hidden[:Decoder_text_glo.n_layers]

        teacher_forcing_ratio = 0.5
        #use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        use_teacher_forcing = False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(tensor_text[1][1][num_batch]):
                detext_output, detext_hidden = Decoder_text_glo(detext_input, detext_hidden)
                loss_text += Criterion_text_glo(detext_output, tensor_text[1][0][num_batch][di])
                detext_input = tensor_text[1][0][num_batch][di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(tensor_text[1][1][num_batch]):
                detext_output, detext_hidden = Decoder_text_glo(detext_input, detext_hidden)
                topv, topi = detext_output.topk(1)
                detext_input = topi.squeeze().detach()  # detach from history as input

                loss_text += Criterion_text_glo(detext_output, tensor_text[1][0][num_batch][di])
                if detext_input.item() == EOS_token:
                    break

    loss_text.backward()

    EnOptimizer_text_glo.step()
    DeOptimizer_text_glo.step()
    HSTOptimizer_glo.step()

    return loss_text.item()/loss_text_size


"""训练数据集。
train_iters(train_dataloader, n_iters, print_every, plot_every)

Args:
    train_dataloader: 训练样本的dataloader
    n_iters: 训练次数
    print_every: 打印相关信息的间隔
    plot_every: 展示loss变化的间隔

******************************
Creat:@ZJianbo @2018.10.15
Update:
"""
def train_iters(train_dataloader, n_iters=10, print_every=100, plot_every=10):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for i_iter in range(1, n_iters + 1):
        for num, training_data in enumerate(train_dataloader):
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
            evaluate_randomly(Encoder_text_glo, Decoder_text_glo, Encoder_HST_glo, trainData, n=1)

        if i_iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    print(plot_losses)
    #showPlot(plot_losses)


"""数据集某一条数据的测试
跟训练时一样的步骤，但是不需要反向传播，只是用来测试
@ZJianbo @2018.10.15
"""
def evaluate(encoder, decoder, enhistory, query, hstsentence):
    input_tensor, input_size = Batch2Tensor().tensor_from_sentence(allDataWords, query)

    entextHST_hidden = []
    for sentence in hstsentence:
        entext_hidden = encoder.init_hidden()
        input_hst_tensor, input_hst_size = Batch2Tensor().tensor_from_sentence(allDataWords, sentence)
        for ei in range(input_hst_size):
            entext_output, entext_hidden = encoder(input_hst_tensor[ei], entext_hidden)
        entextHST_hidden.append(entext_hidden[:decoder.n_layers])

    history_hidden = enhistory.init_hidden()
    len_hst = len(hstsentence)
    for i in range(len_hst):
        history_output, history_hidden = enhistory(entextHST_hidden[len_hst-1-i], history_hidden)
    encoder_hidden = encoder.init_hidden()

    for ei in range(input_size):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    detext_input = torch.tensor([[SOS_token]], device=device)  # SOS
    detext_hidden = encoder_hidden[:decoder.n_layers]+history_hidden[:decoder.n_layers]

    decoded_words = []

    for di in range(MAX_LENGTH):
        detext_output, detext_hidden = decoder(detext_input, detext_hidden)
        topv, topi = detext_output.data.topk(1)
        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(allDataWords.index2word[topi.item()])

        detext_input = topi.squeeze().detach()

    return decoded_words


"""数据集中随机语句的测试，可选择数目n
@ZJianbo @2018.10.15
"""
def evaluate_randomly(encoder, decoder, enhistory, batches, n=3):
    for i in range(n):
        batch = random.choice(batches)
        print('> ', batch['text'])
        print('= ', batch['text_next'])
        output_words = evaluate(encoder, decoder, enhistory, batch['text'], batch['text_history'])
        output_sentence = ' '.join(output_words)
        print('< ', output_sentence)
        print('')


"""对数据集外未知语句的测试
@ZJianbo @2018.10.15
"""
def test_sentence(encoder, decoder, enhistory, sentences):
    print("***** Test *****")
    hst=[]
    for stc in sentences:
        print('> ', stc)
        output_words = evaluate(encoder, decoder, enhistory, stc, hst)
        output_sentence = ' '.join(output_words)
        print('< ', output_sentence)
        print('')


if __name__ == '__main__':
    # 进行训练，Train_Iters和Print_Every可在配置文件中设置
    train_iters(trainDataloader, n_iters=Train_Iters, print_every=Print_Every, plot_every=Print_Every)
    # 保存模型
    if IsSaveModel:
        torch.save(Encoder_text_glo.state_dict(), 'entext.pkl')
        torch.save(Decoder_text_glo.state_dict(), 'detext.pkl')
        torch.save(Encoder_HST_glo.state_dict(), 'textHST.pkl')
    # 从测试集中随机取n条数据进行测试
    print("***** Training Evaluate *****")
    evaluate_randomly(Encoder_text_glo, Decoder_text_glo, Encoder_HST_glo, trainData, n=1)
    # 从测试集中随机取n条数据进行测试
    print("***** Evaluate *****")
    evaluate_randomly(Encoder_text_glo, Decoder_text_glo, Encoder_HST_glo, testData, n=1)
    # 随机输入语句进行测试，TestSentence可在配置文件中设置
    #test_sentence(Encoder_text_glo, Decoder_text_glo, TestSentence)
