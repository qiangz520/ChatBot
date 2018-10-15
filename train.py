#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : train.py
# @Author    : ZJianbo
# @Date	     : 2018/10/13
# @Function  : 开始训练

import random

from predata import *


"""训练单条数据。
train(input_tensor, target_tensor, input_size, target_size)

Args:
    *_tensor: 输入/输出的tensor类型
    *_size: 输入/输出的语句的实际长度
    use_teacher_forcing: 是否使用“教师强迫”，即在decoder时，
                         以一定概率将target，而不是预测的输出，作为输入

******************************
Creat:@ZJianbo @2018.10.15
Update:
"""
def train(input_tensor, target_tensor, input_size, target_size):
    EncoderOptimizer_glo.zero_grad()
    DecoderOptimizer_glo.zero_grad()

    loss = 0
    encoder_hidden = Encoder_glo.init_hidden()

    for ei in range(input_size):
        encoder_output, encoder_hidden = Encoder_glo(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    #use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_size):
            decoder_output, decoder_hidden = Decoder_glo(decoder_input, decoder_hidden)
            loss += Criterion_glo(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_size):
            decoder_output, decoder_hidden = Decoder_glo(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += Criterion_glo(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token: break

    loss.backward()

    EncoderOptimizer_glo.step()
    DecoderOptimizer_glo.step()

    return loss.item()/target_size


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
def train_iters(train_dataloader, n_iters=10, print_every=100, plot_every=100):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for i_iter in range(1, n_iters + 1):
        for num, training_data in enumerate(train_dataloader):
            input_tensor = training_data[0]
            target_tensor = training_data[1]
            input_size = training_data[2].numpy()
            target_size = training_data[3].numpy()
            # print("inputTensor = ", input_tensor)
            # print("inputSize = ", input_size)

            for num_batch in range(train_dataloader.batch_size):
                loss = train(input_tensor[num_batch], target_tensor[num_batch],
                             input_size[num_batch], target_size[num_batch])
                print_loss_total += loss
                plot_loss_total += loss

        if i_iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (CalculateTime().calc_time(start, i_iter / n_iters),
                                         i_iter, i_iter / n_iters * 100, print_loss_avg))
            evaluate_randomly(Encoder_glo, Decoder_glo, trainText, n=1)

    '''
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    '''
    # showPlot(plot_losses)


"""数据集某一条数据的测试 
@ZJianbo @2018.10.15
"""
def evaluate(encoder, decoder, sentence):
    input_tensor, input_size = Batch2Tensor().tensor_from_sentence(allDataWords, sentence)

    encoder_hidden = encoder.init_hidden()
    for ei in range(input_size):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
    decoder_hidden = encoder_hidden
    decoded_words = []

    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(allDataWords.index2word[topi.item()])

        decoder_input = topi.squeeze().detach()

    return decoded_words


"""数据集中随机语句的测试，可选择数目n
@ZJianbo @2018.10.15
"""
def evaluate_randomly(encoder, decoder, batches, n=3):
    for i in range(n):
        batch = random.choice(batches)
        print('> ', batch['text'])
        print('= ', batch['text_next'])
        output_words= evaluate(encoder, decoder, batch['text'])
        output_sentence = ' '.join(output_words)
        print('< ', output_sentence)
        print('')


"""数据集外未知语句的测试
@ZJianbo @2018.10.15
"""
def test_sentence(encoder, decoder, sentence):
    print("*** Test ***")
    print('> ', sentence)
    output_words = evaluate(encoder, decoder, sentence)
    output_sentence = ' '.join(output_words)
    print('< ', output_sentence)
    print('')


if __name__ == '__main__':
    trainIters = 10
    train_iters(trainDataloader, n_iters=trainIters, print_every=1)
    torch.save(Encoder_glo.state_dict(), 'encoder.pkl')
    torch.save(Decoder_glo.state_dict(), 'decoder.pkl')
    test_sentence(Encoder_glo, Decoder_glo, "hello")

