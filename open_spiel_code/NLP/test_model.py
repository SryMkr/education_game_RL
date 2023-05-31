"""
测试模型效果
"""
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import io

# ---------------------首先做数据预处理的工作，这里肯定不涉及保存模型参数，所以不管模型怎么保存，都需要做数据预处理-----------------------
chinese_tokenizer = get_tokenizer(None)  # None代表以空格作为分词器
english_tokenizer = get_tokenizer(None)  # None代表以空格作为分词器

chinese_file_path = 'cet4_chinese_phonetic.txt'  # 中文和音标文件，以空格分割
english_file_path = 'cet4_letter.txt'  # 英文拼写文件

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 用什么设备运行
checkpoint = torch.load('model_parameters/model_parameters_0.7.pt', map_location=device)  # 加载模型参数


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


split_rate = checkpoint['split_rate']  # 分割训练数据和测试数据的比例


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



BATCH_SIZE = checkpoint['batch_size']  # batch size
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
import torch.nn.functional as F
from torch import Tensor
import math


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

    # 设置一个掩码器，掩盖attention，可以在attention上掩盖(掩盖维度)，也可以在attention加之后再掩盖(直接掩盖权重)
    # def generate_attention_mask(self, batch_size, time_step, attention_dim):
    #     # 生成对角阵，在我的例子下面，掩盖的越小训练的越快越好，设置-1比1好
    #     return torch.triu(torch.ones(batch_size, time_step, attention_dim), diagonal=-1)

    # encoder_output是真的获得了encoder的输出值，shape为torch.Size([32, 128, 128]) [time_dim,batch_size,encoder_dim*2]
    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]  # 获得时间步的大小
        # 对（query）复制，方便直接应用与value的时间步长相同 torch.Size([128, 32, 64])  [batch_size, time_dim, dec_hid_dim]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # 将batch_size放到第一个维度[batch_size,time_dim, encoder_dim*2]
        # torch.Size([128, 32, 8]) energy的tensor
        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))  # 得到了注意力的值
        # 直接加个mask让注意力更加集中于某一个value附近
        # mask = self.generate_attention_mask(energy.shape[0], energy.shape[1],
        #                                     energy.shape[2])  # [batch_size,time_dim,attention_out_dim]
        # masked_energy = energy * mask  # 可以掩盖维度，也可以掩盖权重 看情况
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


# 初始化各种参数，这些参数也可以保存到checkpoint中
INPUT_DIM = len(chinese_vocab)
OUTPUT_DIM = len(english_vocab)
ENC_EMB_DIM = checkpoint['ENC_EMB_DIM']
DEC_EMB_DIM = checkpoint['DEC_EMB_DIM']
ENC_HID_DIM = checkpoint['ENC_HID_DIM']
DEC_HID_DIM = checkpoint['DEC_HID_DIM']
ATTN_DIM = checkpoint['ATTN_DIM']
ENC_DROPOUT = checkpoint['ENC_DROPOUT']
DEC_DROPOUT = checkpoint['DEC_DROPOUT']
# 测试模型不需要clip

# 以下都是实例化对象
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # 计算交叉熵的时候，不计算补齐的数，计算损失函数

# 加载模型参数，加载模型之前需要实例化对象
enc.load_state_dict(checkpoint['encoder_state_dict'])  # 加载encoder
attn.load_state_dict(checkpoint['attention_state_dict'])  # 加载attention
dec.load_state_dict(checkpoint['decoder_state_dict'])  # 加载decoder
model.load_state_dict(checkpoint['model_state_dict'])  # 加载model

# ---------------------------------------模型定义结束，开始训练数据----------------------------------------------------------
import Levenshtein as Levenshtein


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
            # for a in zip(predicted_words_list, real_words_list):
            #     print(a)
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
        return epoch_loss / len(iterator), avg_accuracy, avg_completeness, number_correct_spelling / len(accuracy)


# --------------------------------------------测试模型效果-------------------------------------------------------------------
# 评估模型
test_loss, test_avg_accuracy, test_avg_completeness, ratio_of_correct_spelling = evaluate(model, test_iter, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print(f'| Test Accuracy: {test_avg_accuracy:.3f} | Test Completeness: {test_avg_completeness:7.3f} | Test Correct Ratio: {ratio_of_correct_spelling:7.3f} |')
