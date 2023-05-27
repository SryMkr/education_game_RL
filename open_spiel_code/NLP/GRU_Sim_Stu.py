"""
# 本次实验 是上次的维度*2得到的实验结果
损失记录 500->1, 1000->0.4 1500->0.023, 2000->0.003, 2500->0.001, 3000->0.005 基本上最小值就是0.001，模型的复杂度为1，基本上在2000次就收敛了
训练次数3000 
BATCH_SIZE = 24
tecaher=0.5
clip = 1 
ENC_EMB_DIM = 16
DEC_EMB_DIM = 16
ENC_HID_DIM = 32
DEC_HID_DIM = 32
ATTN_DIM = 4
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0
1: 增加维度有利于收敛 2: teacher focus的效果? 3:dropout的效果?

# 第二次实验 将第一次实验结果的参数的teacher focus改为1，实验证明将teacher focus改为1（直接用真实值训练）收敛的好像更快一点
损失记录 500->0.5, 1000->0.03 1500->0.001, 后面没有再训练，基本上提前至少500次到达了0.001
训练次数3000 
BATCH_SIZE = 24
tecaher=1
clip = 1 
ENC_EMB_DIM = 16
DEC_EMB_DIM = 16
ENC_HID_DIM = 32
DEC_HID_DIM = 32
ATTN_DIM = 4
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0
1: 增加维度有利于收敛 2: teacher focus的效果，如果直接用真实值训练，那么提前500轮达到了收敛 3:dropout的效果?

# 第三次实验 将第二次实验结果的参数的dropout从0改为0.5，基本上和第一次实验差不多 实验证明有dropout收敛的很慢
损失记录 500->0.9, 1000->0.2 1500->0.023, 后面没有再训练，基本上提前至少500次到达了0.001
训练次数3000 
BATCH_SIZE = 24
tecaher=1
clip = 1 
ENC_EMB_DIM = 16
DEC_EMB_DIM = 16
ENC_HID_DIM = 32
DEC_HID_DIM = 32
ATTN_DIM = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
1: 增加维度有利于收敛 2: teacher focus的效果，如果直接用真实值训练，那么提前500轮达到了收敛 3:dropout的效果，有dropout收敛的很慢

# 第4次实验 再次增加一倍的维度，收敛速度更快了
损失记录 500->0.1, 1000->0.002 1091次达到了0.001最小值，模型收敛
训练次数3000 
BATCH_SIZE = 24
tecaher=1
clip = 1 
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0
1: 增加维度有利于收敛 2: teacher focus的效果，如果直接用真实值训练，那么提前500轮达到了收敛 3:dropout的效果，有dropout收敛的很慢
# 第5次实验 再次增加一倍的维度，收敛速度更快了,快了3倍，但是模型训练速度明显慢了很多也不是很慢
损失记录 160->0.1, 250->0.01 520次达到了0.001最小值，模型收敛，又加速了很多
训练次数3000 
BATCH_SIZE = 24
tecaher=1
clip = 1 
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
ENC_HID_DIM = 128
DEC_HID_DIM = 128
ATTN_DIM = 16
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0

1: 增加维度有利于收敛 2: teacher focus的效果，如果直接用真实值训练，那么提前500轮达到了收敛 3:dropout的效果，有dropout收敛的很慢
# 第6次实验 再次增加一倍的维度
损失记录 39->1, 86->0.1 120->0.01 240次达到了0.001最小值，模型收敛，又加速了很多,训练时间的话平均一秒一次，也差不多4分钟
训练次数3000 
BATCH_SIZE = 24
tecaher=1
clip = 1 
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ATTN_DIM = 32
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0

1: 增加维度有利于收敛 2: teacher focus的效果，如果直接用真实值训练，那么提前500轮达到了收敛 3:dropout的效果，有dropout收敛的很慢
1: 增加维度有利于收敛 2: teacher focus的效果，如果直接用真实值训练，那么提前500轮达到了收敛 3:dropout的效果，有dropout收敛的很慢
# 第7次实验 试试增加一倍的batch 48，收敛的比第六次实验慢了
损失记录 57->1, 140->0.1 120->0.01
# 第8次实验 减少一倍的batch 12，两秒一个epoch，比第六次实验收敛的还要快，，但是时间翻倍了，仅仅使得循环次数减少，并没有很大程度上改变收敛速度
# 从时间来看，减少batch_size没必要，需要4分钟，和第六次实验需要的时间差不多
损失记录 24->1, 49->0.1 77->0.01   127->0.001

"""
import os

import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import io
import random
import matplotlib.pyplot as plt

random.seed(1)  # 随机数种子
chinese_tokenizer = get_tokenizer(None)  # 以空格分割每个文件，没有任何的分词器
english_tokenizer = get_tokenizer(None)  # 以空格分割每个文件，没有任何的分词器

chinese_file_path = 'simulate_students/cet4_chinese_phonetic.txt'  # 中文和音标文件
english_file_path = 'simulate_students/cet4_letter.txt'  # 英文拼写文件


# 建立单词和索引一一对应的关系，并且将特殊字符也表示为和索引之间的关系，所以返回值就是一个字典{word:index}
def build_vocab(filepath, tokenizer):  # 建立词库
    counter = Counter()  # 统计{word:frequency}
    with io.open(filepath, encoding="utf8") as f:  # 打开文件
        for string_ in f:
            counter.update(tokenizer(string_))  # 统计{word:frequency}
    # {word:frequency}, 并且再加入特殊字符
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


# 每一个单词都和索引一一对应，就可以用索引代表单词了
chinese_vocab = build_vocab(chinese_file_path, chinese_tokenizer)  # 得到所有汉语和音标，里面包含了每一个{word:index}
english_vocab = build_vocab(english_file_path, english_tokenizer)  # 得到所有的单词 {word:index}


# print(len(chinese_vocab.get_stoi()))  # 查看文件中的所有元素
# print(english_vocab.get_stoi())
# print(len(chinese_vocab))
# print(len(english_vocab))


# 将文件中的每一句话都变为数字，这里还不管一些标识符,将训练集和测试集合分开，按照比例分开
def data_process(filepaths, split_rate):
    raw_chinese_iter = iter(io.open(filepaths[0], encoding="utf8"))  # 返回一个迭代器，保存了文件中的每一行
    raw_english_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_chinese_iter, raw_english_iter):  #
        chinese_tensor_ = torch.tensor([chinese_vocab[token] for token in chinese_tokenizer(raw_de)],
                                       dtype=torch.long)  # 这里得到的每一行句子的单词所对应的索引的向量
        english_tensor_ = torch.tensor([english_vocab[token] for token in english_tokenizer(raw_en)],
                                       dtype=torch.long)
        data.append((chinese_tensor_, english_tensor_))  # 所以data得到的数据就是[（汉语，英语字母）]对应的索引
    # random.shuffle(data)  # 随机打乱顺序，因为字母表中是按照顺序排列的
    train_data = data[: int(len(data) * split_rate)]
    test_data = data[int(len(data) * split_rate):]
    return train_data, test_data


train_filepaths = ['simulate_students/cet4_chinese_phonetic.txt', 'simulate_students/cet4_letter.txt']
train_data, test_data = data_process(train_filepaths, split_rate=0.1)  # 最终还是表示为列表中的元组 源和目标一一对应的索引
# print(len(train_data), len(test_data))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 用什么设备运行

BATCH_SIZE = 24  # 每个batch的大小是128
PAD_IDX = chinese_vocab['<pad>']  # 获得padding的索引，是因为转化成的向量长度可能不匹1
BOS_IDX = chinese_vocab['<bos>']  # 获得开始的字符2
EOS_IDX = chinese_vocab['<eos>']  # 获得结束的字符3

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


#  加入首位开始和结束标识符，输入是列表[（汉语，英文）]，如果长度不匹配需要补0，最终返回的分别是德语和英语索引的的列表列表中嵌套列表
def generate_batch(data_batch):
    chinese_batch, english_batch = [], []
    for (chinese_item, letter_item) in data_batch:
        chinese_batch.append(torch.cat([torch.tensor([BOS_IDX]), chinese_item, torch.tensor([EOS_IDX])], dim=0))
        english_batch.append(torch.cat([torch.tensor([BOS_IDX]), letter_item, torch.tensor([EOS_IDX])], dim=0))
    chinese_batch = pad_sequence(chinese_batch,
                                 padding_value=PAD_IDX)  # 每一个batch都按照了最长的time_step对齐了，[max_time_step,batch_size]
    english_batch = pad_sequence(english_batch, padding_value=PAD_IDX)  # 所以说每个batch的输入的time_dim不一样，但是一个batch中一定一样
    return chinese_batch, english_batch


# [（src_index_sentence,trg_index_sentence）]：生成迭代器，每一组的数据都是[max_time_step,batch_size]
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)

# for a, b, in train_iter:
#     print(a.shape)
#     print(b)

import random
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


# 设置编码器
class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,  # 源语言的词表长度
                 emb_dim: int,  # 需要将每一个token都映射为向量的维度
                 enc_hid_dim: int,  # encoder隐藏层的节点数目
                 dec_hid_dim: int,  # decoder隐藏层的节点数，最后要压缩到和decoder一样的隐藏层的节点数
                 dropout: float):  # 丢弃神经元的比例
        super().__init__()

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
        embedded = self.dropout(self.embedding(src))  # [time_dim,batch_size,emb_dim] [34, 128, 32]
        # [num_layers * num_directions, batch_size, encoder_hidden_size]
        # hidden的最后一层的输出保存了time_dim的所有信息,所以输出只有# [num_layers * num_directions，batch_size, encoder_hidden_size]
        outputs, hidden = self.rnn(embedded)  # hidden_shape: torch.Size([2, 128, 64])由两层每一层有64个节点
        # -2和-1是为了得到双向网络的最后一层的状态，并且合并所以得到的维度是 [batch_size, encoder_hidden_size*2]
        # 也就是说将输入的源语言映射到了新的维度上，所以说整个就是将输入的时间步长重新映射到了隐藏层上，这个结果叫做context
        # hidden_shape: torch.Size([128, 64]) # 本来双链RNN链接以后是128后来经过全连接层变成了64位也就是和decoder层一样的隐藏层节点数
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


# encoder对源输入做了一个总结[batch_size, encoder_hidden_size*2]也就是将每一个输入特征表示成了隐藏层双倍的一个特征，这个就是value，这里没有key
class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,  # encoder层隐藏层的节点数
                 dec_hid_dim: int,  # decoder层隐藏层的节点数
                 attn_dim: int):  # attention的维度
        super().__init__()
        self.enc_hid_dim = enc_hid_dim  # encoder层隐藏层的节点数
        self.dec_hid_dim = dec_hid_dim  # decoder层隐藏层的节点数
        # 注意力机制的输入维度是encoder层隐藏层的节点数+decoder层隐藏层的节点数 （value, query）
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)  # 要将（value，query）做一个映射得到其相关关系

    # encoder_output是真的获得了encoder的输出值，shape为torch.Size([32, 128, 128]) [time_dim,batch_size,encoder_dim*2]
    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]  # 获得时间步的大小
        # 对（query）复制，方便直接应用与value的时间步长相同 torch.Size([128, 32, 64])  [batch_size, time_dim , dec_hid_dim]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # 将batch_size放到第一个维度[batch_size,time_dim,encoder_dim*2]
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
                 dropout: int,
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
                              encoder_outputs: Tensor) -> Tensor:
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
                decoder_hidden: Tensor,  # 这个是encoder的hidden的tensor
                encoder_outputs: Tensor) -> Tuple[Tensor]:
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

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 1) -> Tensor:
        batch_size = src.shape[1]  # 获得batch的大小 [time_dim,batch_size]
        max_len = trg.shape[0]  # 获得目标的输出长度 [time_dim,batch_size]
        trg_vocab_size = self.decoder.output_dim  # 获得目标语言的词表大小
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)  # 创建一个tensor保存每一个batch的输出

        encoder_outputs, hidden = self.encoder(src)  # value，第一个query

        # first input to the decoder is the <sos> token，target的时间步长的第一步总是sos，所以直接获得就可以
        output = trg[0, :]  # 其实就是batch的大小128
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output  # 将预测结果保存起来
            teacher_force = random.random() < teacher_forcing_ratio  # 如果预测错误有一半的概率能够给正确的结果
            top1 = output.max(1)[1]  # 预测的最大可能性的输出结果
            output = (trg[t] if teacher_force else top1)  # 虽然是索引，但是torch已经将其能够对应转换为one-hot
        return outputs


INPUT_DIM = len(chinese_vocab)
OUTPUT_DIM = len(english_vocab)
# 参数设置过大会导致训练的特别慢
# ENC_EMB_DIM = 32
# DEC_EMB_DIM = 32
# ENC_HID_DIM = 64
# DEC_HID_DIM = 64
# ATTN_DIM = 8
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ATTN_DIM = 32
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


# 初始化各种各样的权重，内部已经处理好了，维度了什么的都会自己初始化好所以不用管
def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


# 将所有的权重应用到模型中
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())  # 对所有权重使用的优化器


# 统计需要训练多少个数据
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# PAD_IDX = en_vocab.get_stoi('<pad>')

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # 计算交叉熵的时候，不计算补齐的数，计算损失函数

import math
import time

import Levenshtein as Levenshtein

def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):
    model.train()  # 代表该模型是训练
    epoch_loss = 0
    accuracy = [] # 准确度
    completeness = [] # 完整度
    for _, (src, trg) in enumerate(iterator):  # 之所以第一个参数是索引，后面是一个batch的大小
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()  # 每一个batch都将梯度归0
        output = model(src, trg)  # 读取数据 [time_dim, batch_size, vocab_length]
        predicted_outputs = output.permute(1, 0, 2)  # [batch_size, time_dim, vocab_length]
        # 本身就已经是二维[batch,time_step]
        predicted_indices = torch.argmax(predicted_outputs, dim=2)  # 最后一个维度是预测的概率，取最大值[batch_size, time_dim]
        # 循环数据的每一行，并且将每一行里面的每一个数字转换为字母
        predicted_words_list = []
        for sequence in predicted_indices:  # 得到了每一行数据
            predicted_words = []
            for word_index in sequence:
                sequence_words = english_vocab.lookup_token(word_index.item())
                if sequence_words not in ['<unk>', '<pad>', '<bos>', '<eos>']:
                    predicted_words.append(sequence_words)  # 得到每一行的预测值
            predicted_words_list.append(predicted_words)  # 一个batch的预测结果
        # print('预测值', predicted_words_list)
        # 得到每一行的真实值呢
        real_words_list = []
        real_trg = trg.permute(1, 0)  # [batch_size, time_dim]
        for real_trg_sequence in real_trg:
            real_words = []
            for word_index in real_trg_sequence:
                sequence_words = english_vocab.lookup_token(word_index.item())
                if sequence_words not in ['<unk>', '<pad>', '<bos>', '<eos>']:
                    real_words.append(sequence_words)  # 得到每一行的预测值
            real_words_list.append(real_words)  # 一个batch的真实结果
        # print('真实值', real_words_list)
        # for a in zip(predicted_words_list,real_words_list):
        #     print(a)
        # 拼接单词，开始计算准确度和完整度
        for pred, real in zip(predicted_words_list,real_words_list): # 得到预测和真实的列表
          pred_spelling = ''
          real_spelling = ''
          for pred_letter in pred:
            pred_spelling += pred_letter # 得到预测拼写
          for real_letter in real:
            real_spelling += real_letter # 得到真实的拼写
          pred_completeness = 1- Levenshtein.distance(pred_spelling, real_spelling)/len(real_spelling)
          pred_accuracy = Levenshtein.ratio(pred_spelling, real_spelling)
          accuracy.append(pred_accuracy)
          completeness.append(pred_completeness)
        output = output[1:].view(-1, output.shape[-1])  # 预测的输出[time_dim*batch_size, len(vocab)]
        trg = trg[1:].view(-1)  # 真实的输出，也就是标签的index[time_dim*batch_size]
        loss = criterion(output, trg)  # output是对英语词表中的预测概率，trg是index，得到的结果是这个batch的平均损失
        loss.backward()  # 反向传递
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 将梯度锁在一定的位置，防止梯度爆炸
        optimizer.step()  # 优化器计算
        epoch_loss += loss.item()  # 将每一个batch的平均损失相加
    # 所有batch的损失/batch的总数，得到平均损失
    # 返回准确率
    avg_accuracy = sum(accuracy)/len(accuracy)
    avg_completeness = sum(completeness)/len(completeness)
    print('预测准确度',sum(accuracy)/len(accuracy))
    print('预测完整度',sum(completeness)/len(completeness))
    return epoch_loss / len(iterator), avg_accuracy, avg_completeness


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):
    model.eval()  # 标志该模型为评估模型
    epoch_loss = 0
    accuracy = [] # 准确度
    completeness = [] # 完整度
    with torch.no_grad():  # 评估的时候不计算梯度，所以也就没有归零的说法，和训练的时候一毛一样
      for _, (src, trg) in enumerate(iterator):  # 之所以第一个参数是索引，后面是一个batch的大小
          src, trg = src.to(device), trg.to(device)
          optimizer.zero_grad()  # 每一个batch都将梯度归0
          output = model(src, trg)  # 读取数据 [time_dim, batch_size, vocab_length]
          predicted_outputs = output.permute(1, 0, 2)  # [batch_size, time_dim, vocab_length]
          # 本身就已经是二维[batch,time_step]
          predicted_indices = torch.argmax(predicted_outputs, dim=2)  # 最后一个维度是预测的概率，取最大值[batch_size, time_dim]
          # 循环数据的每一行，并且将每一行里面的每一个数字转换为字母
          predicted_words_list = []
          for sequence in predicted_indices:  # 得到了每一行数据
              predicted_words = []
              for word_index in sequence:
                  sequence_words = english_vocab.lookup_token(word_index.item())
                  if sequence_words not in ['<unk>', '<pad>', '<bos>', '<eos>']:
                      predicted_words.append(sequence_words)  # 得到每一行的预测值
              predicted_words_list.append(predicted_words)  # 一个batch的预测结果
          # print('预测值', predicted_words_list)
          # 得到每一行的真实值呢
          real_words_list = []
          real_trg = trg.permute(1, 0)  # [batch_size, time_dim]
          for real_trg_sequence in real_trg:
              real_words = []
              for word_index in real_trg_sequence:
                  sequence_words = english_vocab.lookup_token(word_index.item())
                  if sequence_words not in ['<unk>', '<pad>', '<bos>', '<eos>']:
                      real_words.append(sequence_words)  # 得到每一行的预测值
              real_words_list.append(real_words)  # 一个batch的真实结果
          # print('真实值', real_words_list)
          # for a in zip(predicted_words_list,real_words_list):
          #     print(a)
          # 拼接单词，开始计算准确度和完整度
          for pred, real in zip(predicted_words_list,real_words_list): # 得到预测和真实的列表
            pred_spelling = ''
            real_spelling = ''
            for pred_letter in pred:
              pred_spelling += pred_letter # 得到预测拼写
            for real_letter in real:
              real_spelling += real_letter # 得到真实的拼写
            pred_completeness = 1- Levenshtein.distance(pred_spelling, real_spelling)/len(real_spelling)
            pred_accuracy = Levenshtein.ratio(pred_spelling, real_spelling)
            accuracy.append(pred_accuracy)
            completeness.append(pred_completeness)
          output = output[1:].view(-1, output.shape[-1])  # 预测的输出[time_dim*batch_size, len(vocab)]
          trg = trg[1:].view(-1)  # 真实的输出，也就是标签的index[time_dim*batch_size]
          loss = criterion(output, trg)  # output是对英语词表中的预测概率，trg是index，得到的结果是这个batch的平均损失
          epoch_loss += loss.item()  # 将每一个batch的平均损失相加
      # 所有batch的损失/batch的总数，得到平均损失
      # 返回准确率
      avg_accuracy = sum(accuracy)/len(accuracy)
      avg_completeness = sum(completeness)/len(completeness)
      print('预测准确度',sum(accuracy)/len(accuracy))
      print('预测完整度',sum(completeness)/len(completeness))
      return epoch_loss / len(iterator), avg_accuracy, avg_completeness


# 这个就是计算了一个时间，输入的单位是秒，输出的也是秒
def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time  # 两个时间差的秒数
    elapsed_mins = int(elapsed_time / 60)  # 将差的秒数转化为分钟数
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))  # 相差的秒数
    return elapsed_mins, elapsed_secs


N_EPOCHS = 1000
CLIP = 1

# best_valid_loss = float('inf')  # 最好是完全没有误差的训练
best_valid_loss = float(0.001)  # 最好是完全没有误差的训练
train_lost_list = []  # 保存每个epoch的损失
avg_accuracy_list = []  # 保存每个epoch的损失
avg_completeness_list = []  # 保存每个epoch的损失
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss,avg_accuracy, avg_completeness = train(model, train_iter, optimizer, criterion, CLIP)
    train_lost_list.append(train_loss)  # 保存损失
    avg_accuracy_list.append(avg_accuracy) # 保存准确度
    avg_completeness_list.append(avg_completeness) # 保存准确度
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    # 模型复杂度就是损失的指数函数，是判断模型好坏的一个标准，当前越小越好
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    if train_loss <= best_valid_loss: # 如果已经到了最小值直接结束
      break
    # 输出准确率
# 在这对损失做个图
plt.figure()
ppl_list = [math.exp(train_loss) for train_loss in train_lost_list]
plt.plot(train_lost_list, 'b', label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig("loss.jpg")
plt.figure()
plt.plot(ppl_list, 'r', label='ppl')
plt.ylabel('ppl')
plt.xlabel('epoch')
plt.legend()
plt.savefig("ppl.jpg")

plt.figure()
ppl_list = [math.exp(train_loss) for train_loss in train_lost_list]
plt.plot(train_lost_list, 'b', label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig("loss.jpg")

plt.figure()
plt.plot(avg_accuracy_list, 'r', label='accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig("accuracy.jpg")

plt.figure()
plt.plot(avg_completeness_list, 'r', label='completeness')
plt.ylabel('completeness')
plt.xlabel('epoch')
plt.legend()
plt.savefig("completeness.jpg")

test_loss,test_avg_accuracy,test_avg_completeness = evaluate(model, test_iter, criterion)  # 评估模型
print(f'| 测试数据的损失: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print('测试数据的准确度',test_avg_accuracy)
print('测试数据的完整度',test_avg_completeness)
