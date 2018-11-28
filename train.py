#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File      : train.py
# @Author    : ZJianbo
# @Date	     : 2018/10/13
# @Function  : 开始训练

import random

from predata import *


"""训练单条数据。
train(input_tensors, target_tensors, input_sizes, target_sizes, batchsize)

Args:
    *_tensor: 输入/输出的tensor类型
    *_size: 输入/输出的语句的实际长度
    use_teacher_forcing: 是否使用“教师强迫”，即在decoder时，
                         以一定概率将target，而不是预测的输出，作为下一个的输入

******************************
Creat:@ZJianbo @2018.10.15
Update:@ZJianbo @2018.10.21 添加了进行batch_size批量训练
Update:
"""


def train(input_tensors, target_tensors, input_sizes, target_sizes, batchsize):
    EnOptimizer_text_glo.zero_grad()
    DeOptimizer_text_glo.zero_grad()

    EnOptimizer_face_glo.zero_grad()
    DeOptimizer_face_glo.zero_grad()
    loss_size = 0
    loss = 0
    for num_batch in range(batchsize):
        input_tensor, target_tensor, input_size, target_size = \
            input_tensors[num_batch], target_tensors[num_batch], input_sizes[num_batch], target_sizes[num_batch]

        loss_size += target_size
        entext_hidden = Encoder_text_glo.init_hidden()
        # 编码得到最后的隐藏层
        for ei in range(input_size):
            entext_output, entext_hidden = Encoder_text_glo(input_tensor[ei], entext_hidden)
            # encoder_output, encoder_hidden = Encoder_text_glo(input_tensor[ei], encoder_hidden)
        detext_input = torch.tensor([[SOS_token]], device=device)
        detext_hidden = entext_hidden[:Decoder_text_glo.n_layers]

        teacher_forcing_ratio = 0.5
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        # use_teacher_forcing = False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_size):
                decoder_output, decoder_hidden = Decoder_text_glo(detext_input, detext_hidden)
                loss += Criterion_text_glo(decoder_output, target_tensor[di])
                detext_input = target_tensor[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_size):
                decoder_output, decoder_hidden = Decoder_text_glo(detext_input, detext_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += Criterion_text_glo(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token: break

    loss.backward()

    EnOptimizer_text_glo.step()
    DeOptimizer_text_glo.step()

    return loss.item()/loss_size


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
            input_tensor = training_data[0]   # query text tensor
            target_tensor = training_data[1]
            input_size = training_data[2].numpy()
            target_size = training_data[3].numpy()

            # history_text_tensors_list = training_data[9].numpy()
            # history_face_tensor = training_data[10]
            #
            # history_text_size_list = training_data[11].numpy()
            # history_face_size = training_data[12].numpy()

            text_loss = train(input_tensor, target_tensor,
                         input_size, target_size, train_dataloader.batch_size)
            # history_text_loss = train()
            print_loss_total += text_loss
            plot_loss_total += text_loss

        if i_iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (CalculateTime().calc_time(start, i_iter / n_iters),
                                         i_iter, i_iter / n_iters * 100, print_loss_avg))
            evaluate_randomly(Encoder_text_glo, Decoder_text_glo, trainData, n=1)

    '''
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    '''
    # showPlot(plot_losses)


"""数据集某一条数据的测试
跟训练时一样的步骤，但是不需要反向传播，只是用来测试
@ZJianbo @2018.10.15
"""
def evaluate(encoder, decoder, sentence):
    input_tensor, input_size = Batch2Tensor().tensor_from_sentence(allDataWords, sentence)

    encoder_hidden = encoder.init_hidden()
    for ei in range(input_size):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
    decoder_hidden = encoder_hidden[:Decoder_text_glo.n_layers]
    decoded_words = []

    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
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


"""对数据集外未知语句的测试
@ZJianbo @2018.10.15
"""
def test_sentence(encoder, decoder, sentences):
    print("***** Test *****")
    for stc in sentences:
        print('> ', stc)
        output_words = evaluate(encoder, decoder, stc)
        output_sentence = ' '.join(output_words)
        print('< ', output_sentence)
        print('')


if __name__ == '__main__':
    # 进行训练，Train_Iters和Print_Every可在配置文件中设置
    train_iters(trainDataloader, n_iters=Train_Iters, print_every=Print_Every)
    # 保存模型
    # torch.save(Encoder_text_glo.state_dict(), 'entext.pkl')
    # torch.save(Decoder_text_glo.state_dict(), 'detext.pkl')
    # torch.save(Encoder_face_glo.state_dict(), 'enface.pkl')
    # torch.save(Decoder_face_glo.state_dict(), 'deface.pkl')
    # 从测试集中随机取n条数据进行测试
    print("***** Evaluate *****")
    evaluate_randomly(Encoder_text_glo, Decoder_text_glo, testData, n=3)
    # 随机输入语句进行测试，TestSentence可在配置文件中设置
    test_sentence(Encoder_text_glo, Decoder_text_glo, TestSentence)
