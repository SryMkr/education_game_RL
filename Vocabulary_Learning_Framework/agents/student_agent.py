"""
Simulate two types of students
1: perfect students：can give the perfect spelling, or maximum probability answer
2: forget students: perfect first and add noise (forgetting)

input: condition, answer length, legal actions/available letter, accuracy
output: actions [the index of letter]

Empirical conclusion：
（1）if n > the length of words, the ngrams will be empty. Therefore, we need pad the length.
（2）the advantage of padding: can increase the n then try large n
# 考虑是每次输出最大的可能，还是输出多种可能的结果，最后选择概率最大的那个
# 如何模拟学习的过程，根据反馈信息，更新模型
# 如何根据提供的信息，使得拼写结果更加准确？ 提高拼写结果的准确度

8.15日任务
# 探索不同的n的效果，探索不同的n给出的答案的准确度
# 把整个过程写成一个函数，目前的主要任务是验证n
"""

import os
import random
from typing import List
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook

# ----------------------------------------------read data---------------------------------------------------------------
current_path = os.getcwd()  # get the current path
vocabulary_absolute_path = os.path.join(current_path, 'vocabulary_books', 'CET4',
                                        'Vocab.json')  # get the vocab data path

vocab_instance = ReadVocabBook(vocab_book_path=vocabulary_absolute_path,
                               vocab_book_name='CET4',
                               chinese_setting=False,
                               phonetic_setting=True,
                               POS_setting=False,
                               english_setting=True,
                               )  # initialize vocabulary instance

# read vocab data
vocab_data = vocab_instance.read_vocab_book()  # [[condition, english]......] [['j ɔ r', 'y o u r'], ['j ɝ s ɛ l f', 'y o u r s e l f']]
random.shuffle(vocab_data)  # shuffle the vocabulary book

# get the longest length and shortest length
word_length_list = []
for task in vocab_data:
    word_length_list.append(len(''.join(task[1].split())))
# print(f' the maximum word length is: {max(word_length_list)}, and the shortest word length is: {min(word_length_list)}',)
LONGEST_WORD_LENGTH = max(word_length_list)


training_data = vocab_data[:100]  # training_data
# print(training_data)


# ----------------------------------------------train model-------------------------------------------------------------

def split_information(task: List[str]):
    """ split information into condition, answer, answer_length"""
    condition = task[0]
    answer = task[1]
    answer_length = len(''.join(task[1].split()))
    return condition, answer, answer_length


import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# nltk.download('punkt')


all_tokens = [word_tokenize(data[1]) for data in training_data]  # get all tokens
# print(all_tokens)


# 给每一行的token添加首尾
def generate_ngrams(tokens, n):
    """ generate n-grams with start and end symbol"""
    start = ['<bos>']  # start can be condition
    end = ['<eos>']
    pad = ['<pad>']
    tokens = start + tokens + end
    padding_part = pad * (LONGEST_WORD_LENGTH-len(tokens))
    padding_token = tokens + padding_part
    # print(f' the token list after padding: {padding_token}')
    return list(ngrams(padding_token, n))


_n_length = 8  # test from 1 to 14
_condition_length = _n_length - 1

from nltk.probability import ConditionalFreqDist

def calculate_condition_prob(all_token, n_length, condition_length):
    condition_grams = []
    all_ngrams = [generate_ngrams(token_list, n_length) for token_list in all_token]
    n_grams = [ngram for sublist in all_ngrams for ngram in sublist]
    print("ngrams:", n_grams)
    # 从n-gram中读取开头的几位字母
    for grams in n_grams:
        if list(grams)[0] == '<bos>':
            condition_grams.append(list(grams)[:condition_length])
    print("condition grams from '<bos>':", condition_grams)
    cond_freq_dist = ConditionalFreqDist(
            (tuple(ngram[:condition_length]), ngram[condition_length]) for ngram in n_grams)
    for condition in cond_freq_dist.conditions():
        # print(f"Condition: {condition}")
        for word in cond_freq_dist[condition]:
            freq = cond_freq_dist[condition][word]
            conditional_prob = cond_freq_dist[condition].freq(word)
            # print(f"{word}: {freq}, {conditional_prob}")

    # predict results
    for con_grams in condition_grams:
        for _ in range((LONGEST_WORD_LENGTH-condition_length)):  # 生成 10 个词元
            condition = tuple(con_grams[-condition_length:])
            next_word = cond_freq_dist[condition].max()  # 选择最可能的下一个词元
            con_grams.append(next_word)
        # calculate accuracy, 可是我如何知道当前任务是是么呢？
        print("Generated Text:", " ".join(con_grams))

            # return freq, conditional_prob, condition_grams

calculate_condition_prob(all_tokens, _n_length, _condition_length)
# ----------------------------------------------generate content--------------------------------------------------------
# how to test different n?

# starting_condition = random.choice(list(cond_freq_dist.conditions()))
# # generated_text = ['<s>', 'a', 'c', 'c', 'o']
# generated_text = list(starting_condition)
# print(generated_text)
# for _ in range((LONGEST_WORD_LENGTH-condition_length)):  # 生成 10 个词元
#     condition = tuple(generated_text[-condition_length:])
#     next_word = cond_freq_dist[condition].max()  # 选择最可能的下一个词元
#     if next_word == '<e>':
#         break
#     else:
#         generated_text.append(next_word)
#
# # 输出生成的文本
# print("Generated Text:", " ".join(generated_text))
# 判断准确度




