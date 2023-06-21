"""
train student memory
问题1：一旦训练完成，会导致结果不再发生变化，学生的答案一定是对的
"""
import torch
from torchtext.data.utils import get_tokenizer
import pickle
import random
from typing import Dict
from student_spelling import PositionalEncoding, attn, dec

# -------------------------------------------------load vocab-----------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # define device

tokenizer = get_tokenizer(None)  # None means the tokenizer is 'space'

chinese_vocab_path = 'simulate_student/vocab/chinese_vocab.pkl'  # chinese vocabulary path
english_vocab_path = 'simulate_student/vocab/english_vocab.pkl'  # english vocabulary path

with open(chinese_vocab_path, 'rb') as f:
    chinese_vocab = pickle.load(f)  # download chinese vocab
with open(english_vocab_path, 'rb') as f:
    english_vocab = pickle.load(f)  # download english vocab


# print(english_vocab.get_stoi())  # {word:index}


# convert token to index
def data_process(task_dict: Dict[str, str]):
    data = []
    for chinese_phonetic, raw_english in task_dict.items():
        chinese_tensor_ = torch.tensor([chinese_vocab[token] for token in tokenizer(chinese_phonetic)],
                                       dtype=torch.long)
        english_tensor_ = torch.tensor([english_vocab[token] for token in tokenizer(raw_english)],
                                       dtype=torch.long)
        data.append((chinese_tensor_, english_tensor_))  # 得到输入和输出对应的索引 [(汉语索引，英语索引)]
    return data


# special symbols
PAD_IDX = chinese_vocab['<pad>']  # 1
BOS_IDX = chinese_vocab['<bos>']  # 2
EOS_IDX = chinese_vocab['<eos>']  # 3
UNK_IDX = chinese_vocab['<unk>']  # 0

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


#  加入首位开始和结束标识符，如果长度不匹配需要补1（补的是batch），最终返回嵌套列表
def generate_batch(data_batch):
    chinese_batch, english_batch = [], []
    for (chinese_item, letter_item) in data_batch:
        chinese_batch.append(torch.cat([torch.tensor([BOS_IDX]), chinese_item, torch.tensor([EOS_IDX])], dim=0))
        english_batch.append(torch.cat([torch.tensor([BOS_IDX]), letter_item, torch.tensor([EOS_IDX])], dim=0))
    chinese_batch = pad_sequence(chinese_batch,
                                 padding_value=PAD_IDX)  # 每一个batch都按照了最长的time_step对齐了，[max_time_step,batch_size]
    english_batch = pad_sequence(english_batch, padding_value=PAD_IDX)  # 所以说每个batch的输入的time_dim不一样，但是一个batch中一定一样
    return chinese_batch, english_batch


# 需求1：我输入一个任务组，可以给我正确的结果
# tasks_pool = {'人的 h j u m ʌ n': 'h u m a n', '谦逊的 h ʌ m b ʌ l': 'h u m b l e'}


INPUT_DIM = len(chinese_vocab)
OUTPUT_DIM = len(english_vocab)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ATTN_DIM = 32
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0

# ---------------------------------------------set train sequence-------------------------------------------------------
import torch.nn as nn
from torch import Tensor
import torch.optim as optim


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
        embedded = self.embedding(src)  # [time_dim,batch_size,emb_dim]
        # 在位置信息加上去之前，将汉语和音标的权重设定为固定值
        position_encoding_embedded = self.position_encoding(embedded)  # 将位置信息加上去
        # [num_layers * num_directions, batch_size, encoder_hidden_size]
        # hidden的最后一层的输出保存了time_dim的所有信息,所以输出只有# [num_layers * num_directions，batch_size, encoder_hidden_size]
        outputs, hidden = self.rnn(position_encoding_embedded)
        # -2和-1是为了得到双向网络的最后一层的状态，并且合并所以得到的维度是 [batch_size, encoder_hidden_size*2]
        # 也就是说将输入的源语言映射到了新的维度上，所以说整个就是将输入的时间步长重新映射到了隐藏层上，这个结果叫做context
        # hidden_shape: [batch_size, decoder_dim] # 本来双链RNN链接以后是128后来经过全连接层变成了64位也就是和decoder层一样的隐藏层节点数
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Seq2SeqTrain(nn.Module):
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
        batch_size = src.shape[1]  # [time_dim,batch_size]
        max_len = trg.shape[0]  # [time_dim,batch_size]
        trg_vocab_size = self.decoder.output_dim  # 获得目标语言的词表大小
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)  # 创建一个tensor保存每一个batch的输出
        encoder_outputs, hidden = self.encoder(src)  # value，第一个query
        output = trg[0, :]  # 其实就是batch的大小
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output  # 将预测结果保存起来
            teacher_force = random.random() < teacher_forcing_ratio  # 如果预测错误有一半的概率能够给正确的结果
            top1 = output.max(1)[1]  # 预测的最大可能性的输出结果
            output = (trg[t] if teacher_force else top1)  # 虽然是索引，但是torch已经将其能够对应转换为one-hot
        return outputs


# 初始化各种各样的权重，内部已经处理好了，维度了什么的都会自己初始化好所以不用管
def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # 计算交叉熵的时候，不计算补齐的数，计算损失函数

import Levenshtein as Levenshtein


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer,
          clip: float,
          ):  # 加载保存点的损失

    model.train()  # 代表该模型是训练
    epoch_loss = 0
    number_correct_spelling = 0  # 记录完成正确拼写的比例
    accuracy = []  # 计算所有训练数据结果的准确度
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
            pred_accuracy = Levenshtein.ratio(pred_spelling, real_spelling)
            # 在这再加一个评价标准，也就是得完全一样才算正确
            if pred_spelling == real_spelling:
                number_correct_spelling += 1  # 完全拼写正确加1
            accuracy.append(pred_accuracy)  # 每一个单词都计算，最后得到的所有训练数据的准确度的列表
        output = output[1:].view(-1, output.shape[-1])  # 预测的输出[time_dim*batch_size, len(vocab)]

        trg = trg[1:].view(-1)  # 真实的输出，也就是标签的index[time_dim*batch_size]

        loss = criterion(output, trg)  # output是对英语词表中的预测概率，trg是index，得到的结果是这个batch的平均损失
        loss.backward()  # 反向传递
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 将梯度锁在一定的位置，防止梯度爆炸
        optimizer.step()  # 优化器计算
        epoch_loss += loss.item()  # 将每一个batch的平均损失相加

    # 每一个epoch结束了输出预测单词和真实单词
    # for a in zip(predicted_words_list, real_words_list):
    #     print(a)
    avg_accuracy = sum(accuracy) / len(accuracy)  # 返回平均准确率
    # print('训练数据预测准确度', sum(accuracy) / len(accuracy))
    # print('训练数据完全拼写正确的比例', number_correct_spelling / len(accuracy))
    # 返回整个数据集的平均损失
    return epoch_loss / len(iterator), avg_accuracy, number_correct_spelling / len(accuracy)


# ---------------------------------------------start train--------------------------------------------------------------
import time
import os

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
# 将所有的权重应用到模型中
memory_model = Seq2SeqTrain(enc, dec, device).to(device)
memory_model.apply(init_weights)


# 这个就是计算了一个时间，输入的单位是秒，输出的也是秒
def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time  # 两个时间差的秒数
    elapsed_mins = int(elapsed_time / 60)  # 将差的秒数转化为分钟数
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))  # 相差的秒数
    return elapsed_mins, elapsed_secs


def train_student(tasks_pool):
    train_data = data_process(tasks_pool)
    train_iter = DataLoader(train_data, batch_size=len(tasks_pool),
                            shuffle=True, collate_fn=generate_batch)
    start_epoch = 0
    N_EPOCHS = 500
    if os.path.exists('Word_Maker_RL/model_parameters/model_parameters_trained_0.1.pt'):
        checkpoint = torch.load(
            'Word_Maker_RL/model_parameters/model_parameters_trained_0.1.pt')  # reload model parameter
        memory_model.load_state_dict(checkpoint['model_state_dict'])  # 加载model
        optimizer = optim.Adam(memory_model.parameters())  # 使用adam优化器
        enc.load_state_dict(checkpoint['encoder_state_dict'])  # 加载encoder
        attn.load_state_dict(checkpoint['attention_state_dict'])  # 加载attention
        dec.load_state_dict(checkpoint['decoder_state_dict'])  # 加载decoder
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
        N_EPOCHS = checkpoint['epoch'] + 200  # 每一次结束以后都训练50次
        start_epoch = checkpoint['epoch']
    CLIP = 1
    split_rate = 0.1
    chinese_weight = 0.95
    phoneme_weight = 0.05
    optimizer = optim.Adam(memory_model.parameters())  # 使用adam优化器
    avg_accuracy_threshold = 0.96  # 用准确度来控制模型什么时候结束
    avg_correct_spelling_threshold = 0.96  # 用准确度来控制模型什么时候结束
    train_lost_list = []  # 保存每个epoch的损失
    avg_accuracy_list = []  # 保存每个epoch的损失
    number_of_correct_spelling_list = []  # 完全拼写正确的单词比例
    for epoch in range(start_epoch, N_EPOCHS):
        start_time = time.time()
        train_loss, avg_accuracy, ratio_of_correct_spelling = train(memory_model, train_iter, optimizer, CLIP)
        train_lost_list.append(train_loss)  # 保存损失
        avg_accuracy_list.append(avg_accuracy)  # 保存准确度
        number_of_correct_spelling_list.append(ratio_of_correct_spelling)  # 保存完全拼写正确的比例
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # 模型复杂度就是损失的指数函数，是判断模型好坏的一个标准，当前越小越好
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        if ratio_of_correct_spelling > avg_correct_spelling_threshold:  # 预测准确度控制循环，完全拼写正确的概率非常重要
            break

    # -------------------------------------------------- 达到结束标准后 保存模型----------------------------------------
    torch.save({
        'epoch': epoch,
        'model_state_dict': memory_model.state_dict(),
        'encoder_state_dict': enc.state_dict(),
        'attention_state_dict': attn.state_dict(),
        'decoder_state_dict': dec.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'split_rate': split_rate,
        'chinese_weight': chinese_weight,
        'phoneme_weight': phoneme_weight,
        'ENC_EMB_DIM': ENC_EMB_DIM,
        'DEC_EMB_DIM': DEC_EMB_DIM,
        'ENC_HID_DIM': ENC_HID_DIM,
        'DEC_HID_DIM': DEC_HID_DIM,
        'ATTN_DIM': ATTN_DIM,
        'ENC_DROPOUT': ENC_DROPOUT,
        'DEC_DROPOUT': DEC_DROPOUT,
        'CLIP': CLIP,
    }, 'Word_Maker_RL/model_parameters/model_parameters_trained_0.1.pt')
    print('-------------------------------model saved---------------------------------')
    from word_maker_state import event
    event.set()


if __name__ == "__main__":
    train_student(train_iter)
