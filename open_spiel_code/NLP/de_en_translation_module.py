# !python -m spacy download en_core_web_sm   # 在colab上运行需要导入分词器模型
# !python -m spacy download de_core_news_sm

import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
import io
import spacy

url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]  # 这里下载的是数据集
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')  # 得先下载下来
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')  # 返回的是个分词器模型


# 建立单词和索引一一对应的关系，并且将特殊字符也表示为和索引之间的关系，所以返回值就是一个字典{word:index}
def build_vocab(filepath, tokenizer):  # 建立词库
    counter = Counter()  # 统计{word:frequency}
    with io.open(filepath, encoding="utf8") as f:  # 打开文件
        for string_ in f:
            counter.update(tokenizer(string_))  # 统计{word:frequency}
    # {word:frequency}, 并且再加入特殊字符
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


# 每一个单词都和索引一一对应，就可以用索引代表单词了
de_vocab = build_vocab(train_filepaths[0], de_tokenizer)  # 得到所有的德语词汇是字典的格式，里面包含了每一个{word:index}
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)  # 得到所有的英语词汇 {word:index}


# print(de_vocab.get_stoi())  # 查看文件中的所有元素
# print(de_vocab['<unk>'])
# print(de_vocab['<pad>'])
# print(de_vocab['<bos>'])
# print(de_vocab['<eos>'])


# 将文件中的每一句话都变为数字，这里还不管一些标识符
def data_process(filepaths):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))  # 返回一个迭代器，保存了文件中的每一行
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):  # 将德语与英语压缩为元素，再读取，得到的就是每一行的英语和德语
        #  即使预处理了 有些单词还是不存在 所以还是要先判断存不存在 然后在转换
        # 用分词器将这一行分开，得到token，然后通过词表这个字典，得到这个token的索引
        de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de) if token in de_vocab],
                                  dtype=torch.long)  # 这里得到的每一行句子的单词所对应的索引的向量
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en) if token in en_vocab],
                                  dtype=torch.long)
        data.append((de_tensor_, en_tensor_))  # 所以data得到的数据就是[（德语句子，英语句子）]对应的索引
    return data


train_data = data_process(train_filepaths)  # 最终还是表示为列表中的元组 源和目标一一对应的索引
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 用什么设备运行

BATCH_SIZE = 128  # 每个batch的大小是128
PAD_IDX = de_vocab['<pad>']  # 获得padding的索引，是因为转化成的向量长度可能不匹1
BOS_IDX = de_vocab['<bos>']  # 获得开始的字符2
EOS_IDX = de_vocab['<eos>']  # 获得结束的字符3

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


#  加入首位开始和结束标识符，输入是列表[（德语索引，英语索引）]，如果长度不匹配需要补0，最终返回的分别是德语和英语索引的的列表列表中嵌套列表
def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)  # 每一个batch都按照了最长的time_step对齐了，[max_time_step,batch_size]
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)  # 所以说每个batch的输入的time_dim不一样，但是一个batch中一定一样
    return de_batch, en_batch


# [（src_index_sentence,trg_index_sentence）]：生成迭代器，每一组的数据都是[max_time_step,batch_size]
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)
# for a,b, in test_iter:
#     print(a.shape)
#     print(b.shape)

import random
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


# 设置编码器
class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,  # 源语言的词表长度，也就是源文件中一共有多少个德语词汇
                 emb_dim: int,  # 需要将每一个token都映射为向量的维度
                 enc_hid_dim: int,  # encoder隐藏层的节点数目
                 dec_hid_dim: int,  # decoder隐藏层的节点数，最后要压缩到和decoder一样的隐藏层的节点数
                 dropout: float):  # 丢弃神经元的比例
        super().__init__()

        self.input_dim = input_dim  # 德语词表，embedding使用
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
                teacher_forcing_ratio: float = 0.5) -> Tensor:
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
            output = (trg[t] if teacher_force else top1) # 虽然是索引，但是torch已经将其能够对应转换为one-hot
        return outputs


INPUT_DIM = len(de_vocab)
OUTPUT_DIM = len(en_vocab)
# 参数设置过大会导致训练的特别慢
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

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


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):
    model.train()  # 代表该模型是训练
    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):  # 之所以第一个参数是索引，后面是一个batch的大小
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()  # 每一个batch都将梯度归0
        output = model(src, trg)  # 读取数据
        output = output[1:].view(-1, output.shape[-1])  # 预测的输出[time_dim*batch_size, len(vocab)]
        trg = trg[1:].view(-1)  # 真实的输出，也就是标签的index[time_dim*batch_size]
        loss = criterion(output, trg)  # output是对英语词表中的预测概率，trg是index，得到的结果是这个batch的平均损失
        loss.backward()  # 反向传递
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 将梯度锁在一定的位置，防止梯度爆炸
        optimizer.step()  # 优化器计算
        epoch_loss += loss.item()  # 将每一个batch的平均损失相加
    # 所有batch的损失/batch的总数，得到平均损失
    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):
    model.eval()  # 标志该模型为评估模型
    epoch_loss = 0
    with torch.no_grad():  # 评估的时候不计算梯度，所以也就没有归零的说法，和训练的时候一毛一样
        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


# 这个就是计算了一个时间，输入的单位是秒，输出的也是秒
def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time  # 两个时间差的秒数
    elapsed_mins = int(elapsed_time / 60)  # 将差的秒数转化为分钟数
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))  # 相差的秒数
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')  # 最好是完全没有误差的训练

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    # 模型复杂度就是损失的指数函数，是判断模型好坏的一个标准，当前越小越好
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iter, criterion)  # 评估模型
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

