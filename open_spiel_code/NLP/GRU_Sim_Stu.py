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
"""
1: 源文件中存在特殊字符还有大小写，现在目标文件全是小写，以及输入文件没有重复意思的汉字
3：保存和加载模型

"""
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import io
import matplotlib.pyplot as plt

# ---------------------首先做数据预处理的工作，这里肯定不涉及保存模型参数，所以不管模型怎么保存，都需要做数据预处理-----------------------
# random.seed(1)  # 随机数种子
chinese_tokenizer = get_tokenizer(None)  # None代表以空格作为分词器
english_tokenizer = get_tokenizer(None)  # None代表以空格作为分词器

chinese_file_path = 'cet4_chinese_phonetic.txt'  # 中文和音标文件，以空格分割
english_file_path = 'cet4_letter.txt'  # 英文拼写文件


# 建立词库 return {word:index}
def build_vocab(filepath, tokenizer):
    counter = Counter()  # 统计器 {word:frequency}
    with io.open(filepath, encoding="utf8") as f:  # 打开文件
        for string_ in f:  # 一行一行的读
            counter.update(tokenizer(string_))  # 更新统计器 {word:frequency}
    # {word:index}, 并且再加入特殊字符 unk: unknown words, pad：补充字符，bos,开始标识符，eos,结束标识符
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


chinese_vocab = build_vocab(chinese_file_path, chinese_tokenizer)  # {word:index}
english_vocab = build_vocab(english_file_path, english_tokenizer)  # {word:index}

# print(len(chinese_vocab.get_stoi()))  # 查看文件中的{word:index}
# print(english_vocab.get_stoi())


split_rate = 0.1  # 分割训练数据和测试数据的比例


# 将文件中的每一句话都变为数字，这里还不管标识符,将训练集和测试集合分开，按照比例分开
def data_process(filepaths, split_rate):
    raw_chinese_iter = iter(io.open(filepaths[0], encoding="utf8"))  # 返回一个迭代器
    raw_english_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_chinese, raw_english) in zip(raw_chinese_iter, raw_english_iter):  # 读取迭代器
        chinese_tensor_ = torch.tensor([chinese_vocab[token] for token in chinese_tokenizer(raw_chinese)],
                                       dtype=torch.long)  # 这里得到的每一行句子的单词所对应的索引的向量
        english_tensor_ = torch.tensor([english_vocab[token] for token in english_tokenizer(raw_english)],
                                       dtype=torch.long)
        data.append((chinese_tensor_, english_tensor_))  # 得到输入和输出对应的索引 [(汉语索引，英语索引)]
    # random.shuffle(data)  # 随机打乱顺序，因为单词表是按照顺序排列的，为了提高数据多样性，不再使用
    train_data = data[: int(len(data) * split_rate)]  # 得到训练数据集
    test_data = data[int(len(data) * split_rate):]  # 得到测试数据集
    return train_data, test_data


train_filepaths = ['cet4_chinese_phonetic.txt', 'cet4_letter.txt']  # 源文件，目标文件
train_data, test_data = data_process(train_filepaths, split_rate=split_rate)  # 训练数据集，测试数据集
# print(len(train_data), len(test_data))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 用什么设备运行

BATCH_SIZE = 24  # batch size
PAD_IDX = chinese_vocab['<pad>']  # 获得padding的索引 1，虽然batch输入的长度不一样，但是按照一个batch中的最大长度对齐输入文本
BOS_IDX = chinese_vocab['<bos>']  # 获得开始的字符2
EOS_IDX = chinese_vocab['<eos>']  # 获得结束的字符3
UNK_IDX = chinese_vocab['<unk>']  # 获得不认识的word的字符0

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


#  加入首位开始和结束标识符，如果长度不匹配需要补1，最终返回嵌套列表
def generate_batch(data_batch):
    chinese_batch, english_batch = [], []
    for (chinese_item, letter_item) in data_batch:
        chinese_batch.append(torch.cat([torch.tensor([BOS_IDX]), chinese_item, torch.tensor([EOS_IDX])], dim=0))
        english_batch.append(torch.cat([torch.tensor([BOS_IDX]), letter_item, torch.tensor([EOS_IDX])], dim=0))
    chinese_batch = pad_sequence(chinese_batch,
                                 padding_value=PAD_IDX)  # 每一个batch都按照了最长的time_step对齐了，[max_time_step,batch_size]
    english_batch = pad_sequence(english_batch, padding_value=PAD_IDX)  # 所以说每个batch的输入的time_dim不一样，但是一个batch中一定一样
    return chinese_batch, english_batch


# [(src_index_sentence,trg_index_sentence)]：生成迭代器，每一组的数据都是[max_time_step,batch_size]
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)

# for a, b, in train_iter:
#     print(a.shape)
#     print(b)

# ---------------------定义模型，也是模型中会保存的参数---------------------------------------------------------------------

import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import math

teacher_focus_ratio = 0.5  # 多大概率拿答案训练模型


# 设置一个掩码器，掩盖attention，可以在attention上掩盖(掩盖维度)，也可以在attention加之后再掩盖(直接掩盖权重)
def generate_attention_mask(batch_size, time_step, attention_dim):
    # 生成对角阵，在我的例子下面，掩盖的越小训练的越快越好，设置-1比1好
    return torch.triu(torch.ones(batch_size, time_step, attention_dim), diagonal=-1)


# 定义位置编码，记录word之间的相对位置 max_len代表了时间步长，只是因为每个batch都变长，所以直接设置一个最大值，然后到时候再切割
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 500):
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
        embedded[:, 0, :] *= 0.95
        # 将第二个维度的剩余行代表音标乘以权重，降低音标的比重
        embedded[:, 1:, :] *= 0.05
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
        # 直接加个mask让注意力更加集中于某一个value附近
        mask = generate_attention_mask(energy.shape[0], energy.shape[1],
                                       energy.shape[2])  # [batch_size,time_dim,attention_out_dim]
        masked_energy = energy * mask  # 可以掩盖维度，也可以掩盖权重 看情况
        attention = torch.sum(masked_energy, dim=2)  # attention: torch.Size([128, 32]) [batch_size, time_step]
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
                teacher_forcing_ratio: float = teacher_focus_ratio) -> Tensor:
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
# 参数设置过大会导致训练的特别慢，设置超参数的时候按照以下比例设置
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
optimizer = optim.Adam(model.parameters())  # 使用adam优化器
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # 计算交叉熵的时候，不计算补齐的数，计算损失函数
# 查看所有超参数的名字和维度，是个字典
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# 统计需要训练多少个超参数
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# ---------------------------------------模型定义结束，开始训练数据----------------------------------------------------------
import Levenshtein as Levenshtein


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):
    model.train()  # 代表该模型是训练
    epoch_loss = 0
    number_correct_spelling = 0  # 记录完成正确拼写的比例
    accuracy = []  # 计算所有训练数据结果的准确度
    completeness = []  # 计算所有训练数据结果的完整度
    for _, (src, trg) in enumerate(iterator):  # 第一个参数是索引，后面是一个batch的大小
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()  # 每一个batch都将梯度归0
        output = model(src, trg)  # 读取数据 [time_dim, batch_size, vocab_length]
        predicted_outputs = output.permute(1, 0, 2)  # 调换维度 [batch_size, time_dim, vocab_length]
        # 本身就已经是二维[batch,time_step]
        predicted_indices = torch.argmax(predicted_outputs, dim=2)  # 最后一个维度是预测的概率，取最大值[batch_size, time_dim]
        # 循环数据的每一行，并且将每一行里面的每一个数字转换为字母，得到模型的预测值
        predicted_words_list = []
        for sequence in predicted_indices:  # 得到了每一行数据
            predicted_words = []
            for word_index in sequence:
                sequence_words = english_vocab.lookup_token(word_index.item())
                if sequence_words == '<eos>':  # 遇到EOS不再保存，代表以及预测结束
                    break
                elif sequence_words not in ['<unk>', '<pad>', '<bos>']:  # 不保存这些信息
                    predicted_words.append(sequence_words)  # 得到每一行的预测值
            predicted_words_list.append(predicted_words)  # 一个batch的预测结果
        # 得到每一行的真实值呢
        real_words_list = []
        real_trg = trg.permute(1, 0)  # [batch_size, time_dim]
        for real_trg_sequence in real_trg:
            real_words = []
            for word_index in real_trg_sequence:
                sequence_words = english_vocab.lookup_token(word_index.item())
                if sequence_words == '<eos>':  # 遇到EOS不再保存
                    break
                elif sequence_words not in ['<unk>', '<pad>', '<bos>']:
                    real_words.append(sequence_words)  # 得到每一行的预测值
            real_words_list.append(real_words)  # 一个batch的真实结果
        # 拼接单词，开始计算准确度和完整度
        for pred, real in zip(predicted_words_list, real_words_list):  # 得到预测和真实的列表
            pred_spelling = ''
            real_spelling = ''
            for pred_letter in pred:
                pred_spelling += pred_letter  # 得到预测拼写
            for real_letter in real:
                real_spelling += real_letter  # 得到真实的拼写
            pred_completeness = 1 - Levenshtein.distance(pred_spelling, real_spelling) / len(real_spelling)
            pred_accuracy = Levenshtein.ratio(pred_spelling, real_spelling)
            # 在这再加一个评价标准，也就是得完全一样才算正确
            if pred_spelling == real_spelling:
                number_correct_spelling += 1  # 完全拼写正确加1
            accuracy.append(pred_accuracy)  # 每一个单词都计算，最后得到的所有训练数据的准确度的列表
            completeness.append(pred_completeness)  # 每一个单词都计算，最后得到的所有训练数据的完整度的列表
        output = output[1:].view(-1, output.shape[-1])  # 预测的输出[time_dim*batch_size, len(vocab)]
        trg = trg[1:].view(-1)  # 真实的输出，也就是标签的index[time_dim*batch_size]
        loss = criterion(output, trg)  # output是对英语词表中的预测概率，trg是index，得到的结果是这个batch的平均损失
        loss.backward()  # 反向传递
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 将梯度锁在一定的位置，防止梯度爆炸
        optimizer.step()  # 优化器计算
        epoch_loss += loss.item()  # 将每一个batch的平均损失相加
    # 每一个epoch结束了输出预测单词和真实单词
    for a in zip(predicted_words_list, real_words_list):
        print(a)
    avg_accuracy = sum(accuracy) / len(accuracy)  # 返回平均准确率
    avg_completeness = sum(completeness) / len(completeness)  # 返回平均拼写完整度
    print('训练数据预测准确度', sum(accuracy) / len(accuracy))
    print('训练数据预测完整度', sum(completeness) / len(completeness))
    print('训练数据完全拼写正确的比例', number_correct_spelling/len(accuracy))
    # 返回整个数据集的平均损失
    return epoch_loss / len(iterator), avg_accuracy, avg_completeness, number_correct_spelling/len(accuracy)


# 和训练函数一模一样，除了不计算梯度，所以没有优化器，也没有clip
def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):
    model.eval()  # 标志该模型为评估模型
    epoch_loss = 0
    accuracy = []  # 准确度
    completeness = []  # 完整度
    number_correct_spelling = 0  # 记录完成正确拼写的比例
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
                    if sequence_words == '<eos>':  # 遇到EOS不再保存
                        break
                    elif sequence_words not in ['<unk>', '<pad>', '<bos>']:
                        predicted_words.append(sequence_words)  # 得到每一行的预测值
                predicted_words_list.append(predicted_words)  # 一个batch的预测结果
            # 得到每一行的真实值呢
            real_words_list = []
            real_trg = trg.permute(1, 0)  # [batch_size, time_dim]
            for real_trg_sequence in real_trg:
                real_words = []
                for word_index in real_trg_sequence:
                    sequence_words = english_vocab.lookup_token(word_index.item())
                    if sequence_words == '<eos>':  # 遇到EOS不再保存
                        break
                    elif sequence_words not in ['<unk>', '<pad>', '<bos>']:
                        real_words.append(sequence_words)  # 得到每一行的预测值
                real_words_list.append(real_words)  # 一个batch的真实结果
            for a in zip(predicted_words_list, real_words_list):
                print(a)
            # 拼接单词，开始计算准确度和完整度
            for pred, real in zip(predicted_words_list, real_words_list):  # 得到预测和真实的列表
                pred_spelling = ''
                real_spelling = ''
                for pred_letter in pred:
                    pred_spelling += pred_letter  # 得到预测拼写
                for real_letter in real:
                    real_spelling += real_letter  # 得到真实的拼写
                pred_completeness = 1 - Levenshtein.distance(pred_spelling, real_spelling) / len(real_spelling)
                pred_accuracy = Levenshtein.ratio(pred_spelling, real_spelling)
                if pred_spelling == real_spelling:
                    number_correct_spelling += 1  # 完全拼写正确加1
                accuracy.append(pred_accuracy)
                completeness.append(pred_completeness)
            output = output[1:].view(-1, output.shape[-1])  # 预测的输出[time_dim*batch_size, len(vocab)]
            trg = trg[1:].view(-1)  # 真实的输出，也就是标签的index[time_dim*batch_size]
            loss = criterion(output, trg)  # output是对英语词表中的预测概率，trg是index，得到的结果是这个batch的平均损失
            epoch_loss += loss.item()  # 将每一个batch的平均损失相加
        # 所有batch的损失/batch的总数，得到平均损失
        # 返回准确率
        avg_accuracy = sum(accuracy) / len(accuracy)
        avg_completeness = sum(completeness) / len(completeness)
        return epoch_loss / len(iterator), avg_accuracy, avg_completeness, number_correct_spelling/len(iterator)


import time


# 这个就是计算了一个时间，输入的单位是秒，输出的也是秒
def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time  # 两个时间差的秒数
    elapsed_mins = int(elapsed_time / 60)  # 将差的秒数转化为分钟数
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))  # 相差的秒数
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float(0.01)  # 控制损失到多少结束
avg_accuracy_threshold = 0.98  # 用准确度来控制模型什么时候结束
avg_correct_spelling_threshold = 0.96  # 用准确度来控制模型什么时候结束
train_lost_list = []  # 保存每个epoch的损失
avg_accuracy_list = []  # 保存每个epoch的损失
avg_completeness_list = []  # 保存每个epoch的损失
number_of_correct_spelling_list = []  # 完全拼写正确的单词比例
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, avg_accuracy, avg_completeness, ratio_of_correct_spelling = train(model, train_iter, optimizer, criterion, CLIP)
    train_lost_list.append(train_loss)  # 保存损失
    avg_accuracy_list.append(avg_accuracy)  # 保存准确度
    avg_completeness_list.append(avg_completeness)  # 保存准确度
    number_of_correct_spelling_list.append(ratio_of_correct_spelling)  # 保存完全拼写正确的比例
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    # 模型复杂度就是损失的指数函数，是判断模型好坏的一个标准，当前越小越好
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    if train_loss <= best_valid_loss:  # 如果已经到了最小值直接结束
        break
    elif avg_accuracy >= avg_accuracy_threshold:  # 预测准确度控制循环
        break
    elif ratio_of_correct_spelling > avg_correct_spelling_threshold:
        break


# 在这对损失做个图
fig, axs = plt.subplots(2, 3)
axs[0, 0].plot(train_lost_list, 'b', label='avg_loss')  # 每个epoch的平均损失
plt.ylabel('avg_loss')
plt.xlabel('epoch')

ppl_list = [math.exp(train_loss) for train_loss in train_lost_list]
plt.figure()
axs[0, 1].plot(ppl_list, 'r', label='avg_ppl')
plt.ylabel('avg_ppl')
plt.xlabel('epoch')

axs[1, 0].plot(avg_accuracy_list, 'r', label='avg_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.figure()
axs[1, 1].plot(avg_completeness_list, 'r', label='avg_completeness')
plt.ylabel('avg_completeness')
plt.xlabel('epoch')


plt.figure()
axs[0, 2].plot(number_of_correct_spelling_list, 'r', label='avg_completeness')
plt.ylabel('avg_completeness')
plt.xlabel('epoch')


fig.savefig("evaluation.jpg")

test_loss, test_avg_accuracy, test_avg_completeness, ratio_of_correct_spelling = evaluate(model, test_iter, criterion)  # 评估模型
print(f'| 测试数据的损失: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print('测试数据的准确度', test_avg_accuracy)
print('测试数据的完整度', test_avg_completeness)
print('完全拼写正确的比例', ratio_of_correct_spelling)


