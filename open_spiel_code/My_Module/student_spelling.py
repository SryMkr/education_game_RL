"""
那么如何学习呢？ 如果将反馈的结果可以加到学生的经验当中呢？,怎么设计训练的问题，应该还是在权重上做文章，可能还涉及到继续训练的问题
破解思路是让可能性最小的先选，或者取差集，选择其他的可能性
"""
import random

import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
import pickle
from typing import List, Dict
from collections import Counter
import random

# -------------------------------------------------自定义函数------------------------------------------------------------
def custom_logical_operator(tensor1, tensor2):
    results = []
    for a, b in zip(tensor1, tensor2):
        if a == 0:  # 代表已经确定这个位置不可能是这个字母
            results.append(0)
        elif a == 1 and b == 0:  # 代表虽然词库中这个字母还可选，但是当前这个位置不可能是这个字母
            results.append(0)
        elif a == 1 and b == 1:  # 代表了这个位置可能是这个字母，而且可以选择
            results.append(1)
    return torch.tensor(results).unsqueeze(0)


# -------------------------------------------------加载词库--------------------------------------------------------------
chinese_tokenizer = get_tokenizer(None)  # None代表以空格作为分词器

chinese_vocab_path = 'simulate_student/vocab/chinese_vocab.pkl'  # 中文词库路径
english_vocab_path = 'simulate_student/vocab/english_vocab.pkl'  # 英文词库路径

with open(chinese_vocab_path, 'rb') as f:
    chinese_vocab = pickle.load(f)  # download chinese vocab
with open(english_vocab_path, 'rb') as f:
    english_vocab = pickle.load(f)  # down english vocab

new_dict = {v: k for k, v in english_vocab.get_stoi().items()}  # [index, word]


# print(english_vocab.get_stoi()) # {word:index}
# print(new_dict)


# convert token to index
def data_process(chinese_phonetic):
    data = []
    chinese_tensor_ = torch.tensor([chinese_vocab[token] for token in chinese_tokenizer(chinese_phonetic)],
                                   dtype=torch.long)
    data.append(chinese_tensor_)
    return data


# test_data = data_process('谦逊的 h ʌ m b ʌ l')  # student input
# print(test_data)


PAD_IDX = chinese_vocab['<pad>']  # 1
BOS_IDX = chinese_vocab['<bos>']  # 2
EOS_IDX = chinese_vocab['<eos>']  # 3
UNK_IDX = chinese_vocab['<unk>']  # 0

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


#  加入首位开始和结束标识符，如果长度不匹配需要补1，最终返回嵌套列表
def generate_batch(data_batch):
    chinese_batch = []
    for chinese_item in data_batch:
        chinese_batch.append(torch.cat([torch.tensor([BOS_IDX]), chinese_item, torch.tensor([EOS_IDX])], dim=0))
    chinese_batch = pad_sequence(chinese_batch)  # 每一个batch都按照了最长的time_step对齐了，[max_time_step,batch_size]
    return chinese_batch


# ---------------------定义模型，也是模型中会保存的参数---------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # define device
checkpoint = torch.load('simulate_student/model_parameters/model_parameters_0.1.pt')  # load model parameter


# 定义位置编码，记录word之间的相对位置 max_len代表了时间步长，只是因为每个batch都变长，所以直接设置一个最大值，然后到时候再切割
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.0, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        position_encoding = torch.zeros(max_len, 1, embedding_dim)
        position_encoding[:, 0, 0::2] = torch.sin(position * div_term)  # 根据奇偶数的embedding dim 的偶数列用sin，奇数列用cos
        position_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encoding', position_encoding)  # 加载到模型中

    # 截取和时间步长一样的维度，一直向后广播，所有的数都加上一个相对位置
    def forward(self, x: Tensor) -> Tensor:
        # x: Tensor [seq_len, batch_size, embedding_dim]``
        x = x + self.position_encoding[:x.size(0)]  # 采用的加法操作
        return x


# 设置编码器
class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,  # 源语言的词表长度
                 emb_dim: int,  # 需要将每一个token都映射为向量的维度
                 enc_hid_dim: int,  # encoder隐藏层的节点数目
                 dec_hid_dim: int,  # decoder隐藏层的节点数，最后要压缩到和decoder一样的隐藏层的节点数
                 dropout: float):  # 丢弃神经元的比例
        super().__init__()
        self.position_encoding = PositionalEncoding(emb_dim, dropout)  # 加入位置信息
        self.input_dim = input_dim  # 汉语词表，embedding使用
        self.emb_dim = emb_dim  # token转换为vector的维度
        self.enc_hid_dim = enc_hid_dim  # encoder隐藏层的节点数目，定义了一层的
        self.dec_hid_dim = dec_hid_dim  # decoder隐藏层的节点数
        self.dropout = dropout  # 丢弃神经元的比例
        # 这一层的输出的维度是[time_dim,emb_dim]
        self.embedding = nn.Embedding(input_dim, emb_dim)  # 将每一个token转换为vector，time dimension 由开始和结束标识符决定
        # GRU  [expected features in the input  features in the hidden state h,双链]
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)  # 使用GRU的模型
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)  # 从encoder到decoder的线性转换，因为是双向的，所以要乘以二
        self.dropout = nn.Dropout(dropout)  # inactive的隐藏层节点数目

    # 所以这里的forward是一步输入
    def forward(self, src: Tensor):  # 这里输入的tensor是[time_dim,batch_size]
        embedded = self.embedding(src)  # [time_dim,batch_size,emb_dim] [34, 128, 32]
        # 将第二个维度的第一行代表汉语乘以权重，提高汉语的比重
        embedded[:, 0, :] *= checkpoint['chinese_weight']
        # 将第二个维度的剩余行代表音标乘以权重，降低音标的比重
        embedded[:, 1:, :] *= checkpoint['phoneme_weight']
        # 在位置信息加上去之前，将汉语和音标的权重设定为固定值
        position_encoding_embedded = self.position_encoding(embedded)  # 将位置信息加上去
        # [num_layers * num_directions, batch_size, encoder_hidden_size]
        # hidden的最后一层的输出保存了time_dim的所有信息,所以输出只有# [num_layers * num_directions，batch_size, encoder_hidden_size]
        outputs, hidden = self.rnn(position_encoding_embedded)  # hidden_shape: torch.Size([2, 128, 64])由两层每一层有64个节点
        # -2和-1是为了得到双向网络的最后一层的状态，并且合并所以得到的维度是 [batch_size, encoder_hidden_size*2]
        # 也就是说将输入的源语言映射到了新的维度上，所以说整个就是将输入的时间步长重新映射到了隐藏层上，这个结果叫做context
        # hidden_shape: torch.Size([128, 64]) # 本来双链RNN链接以后是128后来经过全连接层变成了64位也就是和decoder层一样的隐藏层节点数
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


# encoder的output 就是value，而hidden就是query，并且要进入到decoder计算且每次都和value计算相关度
class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,  # encoder层隐藏层的节点数
                 dec_hid_dim: int,  # decoder层隐藏层的节点数
                 attn_dim: int):  # attention的维度
        super().__init__()
        self.enc_hid_dim = enc_hid_dim  # encoder层隐藏层的节点数
        self.dec_hid_dim = dec_hid_dim  # decoder层隐藏层的节点数
        # 注意力机制的输入维度是encoder层隐藏层的节点数+decoder层隐藏层的节点数 （value, query）-> attention dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)  # 要将（value，query）做一个映射得到其相关关系

    # encoder_output是真的获得了encoder的输出值，shape为torch.Size([32, 128, 128]) [time_dim,batch_size,encoder_dim*2]
    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]  # 获得时间步的大小
        # 对（query）复制，方便直接应用与value的时间步长相同 torch.Size([128, 32, 64])  [batch_size, time_dim, dec_hid_dim]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # 将batch_size放到第一个维度[batch_size,time_dim, encoder_dim*2]
        # torch.Size([128, 32, 8]) energy的tensor
        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))  # 得到了注意力的值
        attention = torch.sum(energy, dim=2)  # attention: torch.Size([128, 32]) [batch_size, time_step]
        # 返回的是每一个时间步的权重
        return F.softmax(attention, dim=1)


# 解码的时候的输入是context以及做出的预测
class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,  # 目标语言的词库大小
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)  # 因为预测的输出是输入，所以也需要编码，并于attention的维度拼接
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)  # attention_dime + emd_dim
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)  # output, hidden(quary)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor):
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)  # [batch_size,1,time_step_weight_shape] torch.Size([128, 1, 30])
        # [time_dim, batch_size, encoder_dim * 2]->[batch_size, time_dim,encoder_dim * 2] torch.Size([128, 30, 128])
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # 这是为了求出每个[batch_size, 1, enc_hid_dim] torch.Size([128, 1, 128])
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        # weighted_encoder_rep2: torch.Size([1, 128, 128])
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self,
                input: Tensor,
                decoder_hidden,  # 这个是encoder的hidden的tensor
                encoder_outputs: Tensor):
        input = input.unsqueeze(0)  # input_shape torch.Size([1, 128])  [1, batch_size]
        embedded = self.dropout(
            self.embedding(input))  # embedded_shape torch.Size([1, 128, 32]) [1, batch_size,emd_dim]
        # weighted_encoder_rep_shape torch.Size([1, 128, 128]),quary与那个time_dim更接近
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)
        # attention_dim + emd_dim
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)  # rnn_input_shape torch.Size([1, 128, 160])
        # output_shape torch.Size([1, 128, 64])
        # decoder_hidden_shape torch.Size([1, 128, 64])
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        # 正式得到了decoder的预测输出 output_shape torch.Size([128, 10838])  10838是英语词表的大小
        # output:输出词与输入词的相关性，输出词，输入词->此表中的概率
        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim=1))
        softmax_output = F.softmax(output, dim=1)  # 将输出改为概率，那么所有的概率都将是大于0小于1的数
        return softmax_output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # ----------------------------------------------拼写规则主要在这里修改-------------------------------------------------
    # mask 代表了学生的记忆 1:对固定的思维复制一个mask，如果前面的位置选择了一个字母，导致后面的位置没有可选字母则这个字母不合法
    def forward(self, src: Tensor, available_letter: List[str], student_feedback: Dict[str, int], masks,
                target_length: int):
        batch_size = src.shape[1]  # [time_dim,batch_size]
        trg_vocab_size = self.decoder.output_dim  # get the target vocab size
        outputs = torch.zeros(target_length, batch_size, trg_vocab_size).to(self.device)  # 多一行unk
        encoder_outputs, hidden = self.encoder(src)  # value，query
        output = src[0, :]  # 第一行永远是‘<unk>’

        available_letter_index_list = []
        for letter in available_letter:  # 循环每一个可选字母
            available_letter_index_list.append(english_vocab.get_stoi()[letter])  # 得到了所有可能的结果
        available_letter_index_counter = dict(Counter(available_letter_index_list))  # [word_index :counter]

        if masks is None:  # 初始化一个masks
            masks = torch.zeros(target_length, batch_size, trg_vocab_size).to(self.device)
            real_available_letter_index_list = []  # 存放可选字母的索引
            for letter in available_letter:  # 循环每一个可选字母
                real_available_letter_index_list.append(english_vocab.get_stoi()[letter])  # 得到了所有可能的结果
            masks[1:, :, real_available_letter_index_list] = 1  # 1的个数就表示了该位置上的字母的可能性
        else:
            masks = masks  # 承接上次的学生的记忆
            for key, value in student_feedback.items():
                current_letter = key.split('_')[0]  # 获得这个字母
                current_index = key.split('_')[1]  # 获得这个字母对应拼写的位置
                current_letter_index = english_vocab.get_stoi()[current_letter]  # 找到这个字母所在的索引
                # 然后根据反馈确定现在的先验知识
                if value == 0:  # 红色
                    masks[0:, :, current_letter_index] = 0  # 红色代表绝对不可能，这一列都变为0
                    available_letter_index_counter[current_letter_index] = 0  # red present counter = 0
                elif value == 1:  # 黄色
                    masks[int(current_index) + 1, :, current_letter_index] = 0  # 黄色代表这个位置不可能
                elif value == 2:  # 绿色
                    masks[int(current_index) + 1, :, :] = 0  # 绿色要把这一行都先标记为不可能
                    masks[int(current_index) + 1, :, current_letter_index] = 1  # 然后再把这个点标记为1
                    available_letter_index_counter[current_letter_index] -= 1  # green present fixed
        available_letter_index_copy = available_letter_index_counter.copy()  # 需要复制一份可用字母，不然会自动减少

        for t in range(1, target_length):  # 需要强制将其缩短为目标长度，不然特殊字符也会做预测
            # 只收集大于0的才表示可以选择的字母
            dynamic_available_letter_index_list = [letter for letter, counts in available_letter_index_copy.items() if
                                                   counts > 0]
            output, hidden = self.decoder(output, hidden, encoder_outputs)  # 输入是一个索引
            mask = torch.zeros((1, trg_vocab_size))  # 生成0的初始化tensor
            mask[0, dynamic_available_letter_index_list] = 1  # 真实可用的字母
            real_mask = custom_logical_operator(masks[t, 0, :], mask[0, :])  # 代表了哪些字母可以被选择
            output = output * real_mask  # 将mask和output相乘,得到只能接受的字母索引(最后可能是全0的情况)
            # indices = torch.where((output[0, :] > 0))[0].tolist()

            if torch.all(real_mask == 0).item():  # 如果output全零，那么选择mask中等于1的那个索引的字母
                if (masks[t] == 1).sum() == 1:  # 代表它是确定性的字母
                    top1 = masks[t].max(1)[1]
                    output = masks[t]
                else:
                    top1 = torch.tensor([random.choice(dynamic_available_letter_index_list)])
                    new_output = torch.zeros((1, trg_vocab_size))  # 生成0的初始化tensor
                    new_output[0, top1.item()] = 0.5  # 真实可用的字母
                    output = new_output

            else:
                top1 = output.max(1)[1]  # 预测的最大可能性的输出结果
            outputs[t] = output  # 将预测结果保存起来

            # print('可选择的字母是：', [new_dict[index] for index in indices])
            # print('词库中的字母是：', [new_dict[index] for index in dynamic_available_letter_index_list])
            # print('可选择的字母的个数是：', [available_letter_index_copy[index] for index in indices])
            # print('选择的字母是：', top1.item(), new_dict[top1.item()])
            # 完全确定的字母已经再之前就确定好了，所以没必要再减，只有不确定的才需要减去
            if (masks[t] == 1).sum() != 1:
                available_letter_index_copy[top1.item()] -= 1  # 将该字母的counts -1
            output = top1  # 将预测结果作为下一轮的输入
        return outputs, masks


# # 初始化各种参数，这些参数也可以保存到checkpoint中
INPUT_DIM = len(chinese_vocab)
OUTPUT_DIM = len(english_vocab)
ENC_EMB_DIM = checkpoint['ENC_EMB_DIM']
DEC_EMB_DIM = checkpoint['DEC_EMB_DIM']
ENC_HID_DIM = checkpoint['ENC_HID_DIM']
DEC_HID_DIM = checkpoint['DEC_HID_DIM']
ATTN_DIM = checkpoint['ATTN_DIM']
ENC_DROPOUT = checkpoint['ENC_DROPOUT']
DEC_DROPOUT = checkpoint['DEC_DROPOUT']

# # 以下都是实例化对象
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

# download parameters
enc.load_state_dict(checkpoint['encoder_state_dict'])  # 加载encoder
attn.load_state_dict(checkpoint['attention_state_dict'])  # 加载attention
dec.load_state_dict(checkpoint['decoder_state_dict'])  # 加载decoder
model.load_state_dict(checkpoint['model_state_dict'])  # 加载model


# ---------------------------------------模型定义结束，开始按照规则拼写单词----------------------------------------------------------
def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             available_letter: List[str],
             student_feedback: List[int],
             masks,
             target_length):
    model.eval()  # evaluation mode
    with torch.no_grad():  # no grad
        for _, src in enumerate(iterator):  # simulate the student to see the chinese and phonetic
            src = src.to(device)  # set the device
            output, masks = model(src, available_letter, student_feedback, masks, target_length)  # spell the word
            predicted_outputs = output.permute(1, 0, 2)  # [batch_size, time_dim, vocab_length]
            # 本身就已经是二维[batch,time_step]
            predicted_indices = torch.argmax(predicted_outputs, dim=2)  # 最后一个维度是预测的概率，取最大值[batch_size, time_dim]
            # 循环数据的每一行，并且将每一行里面的每一个数字转换为字母
            predicted_words_list = []
            for sequence in predicted_indices:  # 得到了每一行数据
                predicted_words = []
                for word_index in sequence:
                    sequence_words = english_vocab.lookup_token(word_index.item())
                    if sequence_words == '<eos>':  # 遇到EOS不再保存
                        break
                    elif sequence_words not in ['<unk>', '<pad>', '<bos>']:
                        predicted_words.append(sequence_words)  # 得到每一行的预测值
                predicted_words_list.append(predicted_words)  # 一个batch的预测结果
            # 拼接单词，开始计算准确度和完整度
            for pred in predicted_words_list:  # 得到预测和真实的列表
                pred_spelling = ''
                for pred_letter in pred:
                    pred_spelling += pred_letter  # 得到预测拼写
                    pred_spelling += ' '  # 中间以空格分割
        return pred_spelling, masks
