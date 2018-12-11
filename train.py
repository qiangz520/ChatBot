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
    EnOptimizer_face_glo.zero_grad()
    DeOptimizer_face_glo.zero_grad()

    loss_text_size = 0
    loss_face_size = 0
    loss_text = 0
    loss_face = 0

    for num_batch in range(batchsize):
        loss_text_size += tensor_text[1][1][num_batch].numpy()
        loss_face_size += tensor_face[1][1][num_batch].numpy()

        # facs_prev
        enface_hidden = Encoder_face_glo.init_hidden()
        for ei in range(tensor_face[2][1][num_batch]):
            enface_output, enface_hidden = Encoder_face_glo(tensor_face[2][0][num_batch][ei], enface_hidden)
        enface_hidden = enface_hidden[:Decoder_face_glo.n_layers]

        #   <HST>
        entextHST_hidden = []
        for i in range(10):
            entext_hidden = Encoder_text_glo.init_hidden()
            for ei in range(tensor_text[i+2][1][num_batch]):
                entext_output, entext_hidden = Encoder_text_glo(tensor_text[i+2][0][num_batch][ei], entext_hidden)
            entextHST_hidden.append(entext_hidden[:Decoder_text_glo.n_layers])

        history_hidden = Encoder_HST_glo.init_hidden()
        for i in range(10):
            if i==9:
                history_input = torch.cat((entextHST_hidden[9-i], enface_hidden), 1)
            else:
                history_input = torch.cat((torch.tensor(entextHST_hidden[9-i],device=device).float(), torch.tensor([[np.zeros(hiddenSize)]],device=device).float()), 1)
            history_output, history_hidden = Encoder_HST_glo(history_input, history_hidden)
        #   </HST>

        # text_query
        entext_hidden = Encoder_text_glo.init_hidden()
        for ei in range(tensor_text[0][1][num_batch]):
            entext_output, entext_hidden = Encoder_text_glo(tensor_text[0][0][num_batch][ei], entext_hidden)
        # face_query
        enface_hidden = Encoder_face_glo.init_hidden()
        for ei in range(tensor_face[0][1][num_batch]):
            enface_output, enface_hidden = Encoder_face_glo(tensor_face[0][0][num_batch][ei], enface_hidden)

        detext_input = torch.tensor([[SOS_token]], device=device)
        deface_input = torch.tensor([[0]], device=device)
        decoder_hidden = entext_hidden[:Decoder_text_glo.n_layers] + enface_hidden[:Decoder_face_glo.n_layers] \
                        + history_hidden[:Decoder_text_glo.n_layers]
        deface_hidden = decoder_hidden[:]
        detext_hidden = decoder_hidden[:]
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

            for di in range(tensor_face[1][1][num_batch]):
                deface_output, deface_hidden = Decoder_face_glo(deface_input, deface_hidden)
                topv, topi = deface_output.topk(1)
                deface_input = topi.squeeze().detach()  # detach from history as input

                loss_face += Criterion_face_glo(deface_output, tensor_face[1][0][num_batch][di])
                if deface_input.item() == 1:
                    break

    loss_text.backward(retain_graph=True)
    loss_face.backward()

    EnOptimizer_text_glo.step()
    DeOptimizer_text_glo.step()
    HSTOptimizer_glo.step()
    EnOptimizer_face_glo.step()
    DeOptimizer_face_glo.step()

    return loss_text.item()/loss_text_size, loss_face.item()/loss_face_size


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

            loss_text, loss_face = train(tensor_text, tensor_face, train_dataloader.batch_size)
            loss = loss_text+loss_face
            print_loss_total += loss
            plot_loss_total += loss

        if i_iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (CalculateTime().calc_time(start, i_iter / n_iters),
                                         i_iter, i_iter / n_iters * 100, print_loss_avg))
            evaluate_randomly(Encoder_text_glo, Decoder_text_glo, Encoder_face_glo,
                              Decoder_face_glo, Encoder_HST_glo, trainData, n=1)

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
def evaluate(encoderText, decoderText, encoderFace, decoderFace, enhistory, data):
    query_text, query_face = data["text"], data["facs"]
    history_text, prev_face = data["text_history"], data["facs_prev"]

    text_tensor, text_size = Batch2Tensor().tensor_from_sentence(allDataWords, query_text)
    face_tensor, face_size = Batch2Tensor().tensor_from_faces(allDataFaces, query_face)
    prevface_tensor, prevface_size = Batch2Tensor().tensor_from_faces(allDataFaces, prev_face)

    entextHST_hidden = []
    for sentence in history_text:
        entext_hidden = encoderText.init_hidden()
        input_hst_tensor, input_hst_size = Batch2Tensor().tensor_from_sentence(allDataWords, sentence)
        for ei in range(input_hst_size):
            entext_output, entext_hidden = encoderText(input_hst_tensor[ei], entext_hidden)
        entextHST_hidden.append(entext_hidden[:decoderText.n_layers])

    enface_hidden = encoderFace.init_hidden()
    for ei in range(prevface_size):
        enface_output, enface_hidden = encoderFace(prevface_tensor[ei], enface_hidden)
    enface_hidden = enface_hidden[:decoderFace.n_layers]


    history_hidden = enhistory.init_hidden()
    len_hst = len(history_text)
    for i in range(len_hst):
        if i == len_hst-1:
            history_input = torch.cat((entextHST_hidden[len_hst-1-i], enface_hidden), 1)
        else:
            history_input = torch.cat((torch.tensor(entextHST_hidden[len_hst-1-i], device=device).float(),
                                       torch.tensor([[np.zeros(hiddenSize)]], device=device).float()), 1)
        history_output, history_hidden = enhistory(history_input, history_hidden)

    entext_hidden = encoderText.init_hidden()
    for ei in range(text_size):
        entext_output, entext_hidden = encoderText(text_tensor[ei], entext_hidden)

    enface_hidden = encoderFace.init_hidden()
    for ei in range(face_size):
        enface_output, enface_hidden = encoderFace(face_tensor[ei], enface_hidden)

    decoder_hidden = entext_hidden[:decoderText.n_layers]+enface_hidden[:decoderFace.n_layers]\
                    + history_hidden[:decoderText.n_layers]
    deface_hidden = decoder_hidden[:]
    detext_hidden = decoder_hidden[:]

    detext_input = torch.tensor([[SOS_token]], device=device)
    deface_input = torch.tensor([[0]], device=device)

    decoder_words = []
    for di in range(MAX_LENGTH):
        detext_output, detext_hidden = decoderText(detext_input, detext_hidden)
        topv, topi = detext_output.data.topk(1)
        if topi.item() == EOS_token:
            decoder_words.append('<EOS>')
            break
        else:
            decoder_words.append(allDataWords.index2word[topi.item()])
        detext_input = topi.squeeze().detach()

    decoder_facs = []
    for di in range(MAX_LENGTH):
        deface_output, deface_hidden = decoderFace(deface_input, deface_hidden)
        topv, topi = deface_output.topk(1)
        deface_input = topi.squeeze().detach()  # detach from history as input
        if deface_input.item() == 1:
            decoder_facs.append('<EOS>')
            break
        else:
            decoder_facs.append(topi.item())
        deface_input = topi.squeeze().detach()

    return decoder_words, decoder_facs


"""数据集中随机语句的测试，可选择数目n
@ZJianbo @2018.10.15
"""
def evaluate_randomly(encoderText, decoderText, encoderFace, decoderFace, enhistory, batches, n=3):
    for i in range(n):
        batch = random.choice(batches)
        words, facs = evaluate(encoderText, decoderText, encoderFace, decoderFace, enhistory, batch)
        output_sentence = ' '.join(words)
        print('> ', batch['text'])
        print('= ', batch['text_next'])
        print('< ', output_sentence)
        print('------------')
        print('> ', allDataFaces.get_faces_type(batch['facs']))
        print('= ', allDataFaces.get_faces_type(batch['facs_next']))
        print('< ', facs)
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
        torch.save(Encoder_text_glo.state_dict(), './ModelPKL/enface12.pkl')
        torch.save(Decoder_text_glo.state_dict(), './ModelPKL/deface12.pkl')
        torch.save(Encoder_face_glo.state_dict(), './ModelPKL/enface12.pkl')
        torch.save(Decoder_face_glo.state_dict(), './ModelPKL/deface12.pkl')
        torch.save(Encoder_HST_glo.state_dict(), './ModelPKL/textHST12.pkl')

    # 从测试集中随机取n条数据进行测试
    print("***** Training Evaluate *****")
    evaluate_randomly(Encoder_text_glo, Decoder_text_glo, Encoder_face_glo,
                      Decoder_face_glo, Encoder_HST_glo, trainData, n=1)
    # 从测试集中随机取n条数据进行测试
    print("***** Evaluate *****")
    evaluate_randomly(Encoder_text_glo, Decoder_text_glo, Encoder_face_glo,
                      Decoder_face_glo, Encoder_HST_glo, testData, n=1)
    # 随机输入语句进行测试，TestSentence可在配置文件中设置
    # #test_sentence(Encoder_text_glo, Decoder_text_glo, TestSentence)
