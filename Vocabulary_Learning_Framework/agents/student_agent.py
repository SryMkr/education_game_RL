"""
Simulate two types of students
1: perfect students：can give the perfect spelling, or maximum probability answer
2: forget students: perfect first and add noise (forgetting)

input: condition, answer length, legal actions/available letter, accuracy
output: actions [the index of letter]

Empirical conclusion：
（1）if n > the length of words, the ngrams will be empty. Therefore, we need pad the length.
（2）the advantage of padding: can increase the n then try large n
（3）the best n is relative to the size of training data

# 如何模拟学习的过程，根据反馈信息，更新模型（如何更新模型的问题）
# 如何模拟遗忘的问题，在频率上加噪声该怎么实现，是拼写的时候添加噪声，还是记忆的时候添加噪声？

8.15日任务
# 如何根据提供的信息（音标什么的），使得拼写结果更加准确？ 提高拼写结果的准确度
"""

import os
import random
from typing import List
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import ConditionalFreqDist
import Levenshtein
import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------------------------read vocabulary data---------------------------------------------------------------
current_path = os.getcwd()  # get the current path
# get the vocab data path
vocabulary_absolute_path = os.path.join(current_path, 'vocabulary_books', 'CET4', 'Vocab.json')
# initialize vocabulary instance
vocab_instance = ReadVocabBook(vocab_book_path=vocabulary_absolute_path,
                               vocab_book_name='CET4',
                               chinese_setting=False,
                               phonetic_setting=True,
                               POS_setting=False,
                               english_setting=True)

# read vocab data [[condition, english]......] [['j ɔ r', 'y o u r'], ['j ɝ s ɛ l f', 'y o u r s e l f']]
vocab_data: List[List[str]] = vocab_instance.read_vocab_book()
# get the longest length and shortest length
word_length_list = []
for task in vocab_data:
    word_length_list.append(len(''.join(task[1].split())))
print(f' the maximum word length is: {max(word_length_list)}, and the shortest word length is: {min(word_length_list)}',)
LONGEST_WORD_LENGTH = max(word_length_list)  # 14

random.shuffle(vocab_data)  # shuffle the vocabulary book


# ----------------------------------------------train model-------------------------------------------------------------

# nltk.download('punkt') down the necessary file for tokenize


def generate_ngrams(tokens, n):
    """ generate n-grams with start, end and pad symbol"""
    start = ['<bos>']  # start can be information
    end = ['<eos>']
    pad = ['<pad>']
    tokens = start + tokens + end
    padding_part = pad * (LONGEST_WORD_LENGTH + 2 - len(tokens))
    padding_token = tokens + padding_part
    # print(f' the token list after padding: {padding_token}')
    return list(ngrams(padding_token, n))


def calculate_condition_prob(training_data, all_token, n_length, condition_length):
    # --------------------------------------calculate probability matrix------------------------------------------------
    condition_grams_with_bos = []  # the condition must has '<bos>' symbol
    generated_answer = []  # the generated content via probability matrix
    all_ngrams = [generate_ngrams(token_list, n_length) for token_list in all_token]
    n_grams = [ngram for sublist in all_ngrams for ngram in sublist]
    # print("ngrams:", n_grams)
    # read the condition with '<bos>'
    for grams in n_grams:
        if list(grams)[0] == '<bos>':
            condition_grams_with_bos.append(list(grams)[:condition_length])
    # print("condition grams from '<bos>':", condition_grams)
    # print(f' the length of conditional grams with <bos>: {len(condition_grams_with_bos)}')
    cond_freq_dist = ConditionalFreqDist(
        (tuple(ngram[:condition_length]), ngram[condition_length]) for ngram in n_grams)
    # for condition in cond_freq_dist.conditions(): # 加噪声的时候可能用的到，因为噪声加载了频率上
    #     # print(f"Condition: {condition}")
    #     for word in cond_freq_dist[condition]:
    #         freq = cond_freq_dist[condition][word]
    #         conditional_prob = cond_freq_dist[condition].freq(word)
    #         # print(f"{word}: {freq}, {conditional_prob}")

    # --------------------------------------predict results----------------------------------------------------------
    word_index = 0
    for con_grams in condition_grams_with_bos:
        for _ in range(LONGEST_WORD_LENGTH + 1 - condition_length):
            try:
                condition = tuple(con_grams[-condition_length:])
                next_word = cond_freq_dist[condition].max()  # select the maximum probability
                con_grams.append(next_word)
            except ValueError as e:
                print(f"Error: {e}")
                print(f"No samples for '{condition}'. Add some samples first.")

        # only read the legal letters
        pure_con_grams = [letter for letter in con_grams if letter not in ['<bos>', '<eos>', '<pad>']]
        generated_answer.append(" ".join(pure_con_grams))
        word_index += 1
        # print("Generated Text:", generated_answer)

    # ----------------------------calculate accuracy and completeness--------------------------------------------------
    accuracy_list = []
    completeness_list = []
    for index in range(len(training_data)):
        accuracy = round(
            Levenshtein.ratio(''.join(generated_answer[index].split(' ')), ''.join(training_data[index][1].split(' '))),
            2)
        completeness = round(1 - Levenshtein.distance(''.join(generated_answer[index].split(' ')),
                                                      ''.join(training_data[index][1].split(' '))) / word_length_list[
                                 index], 2)
        # if accuracy != 1:  # find the wrong spelling
        #     print(f'n={n_length}', ''.join(generated_answer[index].split(' ')), ''.join(training_data[index][1].split(' ')))
        accuracy_list.append(accuracy)
        completeness_list.append(completeness)

    avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    avg_completeness = sum(completeness_list) / len(completeness_list)
    return avg_accuracy, avg_completeness


def draw_n_plots(data_size: int):
    """explore the relationship between n and vocabulary size"""
    training_data = vocab_data[:data_size]  # training_data
    all_tokens = [word_tokenize(data[1]) for data in training_data]  # get all tokens
    _n_length = [n_length for n_length in range(2, LONGEST_WORD_LENGTH + 2, 1)]  # test from 2 to 14 [2-14]
    # print(_n_length)
    _condition_length = [con_length - 1 for con_length in _n_length]
    # print(_condition_length)
    n_accuracy = []
    n_completeness = []

    for n_index in range(len(_n_length)):
        avg_accuracy, avg_completeness = calculate_condition_prob(training_data, all_tokens, _n_length[n_index],
                                                                  _condition_length[n_index])
        n_accuracy.append(avg_accuracy)
        n_completeness.append(avg_completeness)

    # plt.plot(_n_length, n_accuracy, label='n_accuracy', color='blue')
    # plt.plot(_n_length, n_completeness, label='n_completeness', color='red')
    # plt.show()
    return n_accuracy, n_completeness


if __name__ == '__main__':
    _data_size = [size for size in range(50, 3050, 100)]
    _n_length = [n_length for n_length in range(2, LONGEST_WORD_LENGTH + 2, 1)]  # test from 2 to 14 [2-14]
    accuracy = []
    completeness = []
    for size in _data_size:
        avg_accuracy, avg_completeness = draw_n_plots(size)
        print(f'data size:{size}, avg_accuracy:{avg_accuracy}, avg_completeness:{avg_completeness}')
        accuracy.append(avg_accuracy)
        completeness.append(avg_completeness)

    acc_df = pd.DataFrame(accuracy, columns=_n_length, index=_data_size)
    com_df = pd.DataFrame(completeness, columns=_n_length, index=_data_size)
    # 将 DataFrame 保存为 CSV 文件
    acc_df.to_excel('test_data/acc_data.xls')
    com_df.to_excel('test_data/com_data.xls')
