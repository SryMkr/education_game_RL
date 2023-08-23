"""
1: 没有利用任何别的信息，导致预测得准确度不高
2：会出现没有样本的问题，怎么处理？如果直接将模型用于测试信息全都是没出现过的情况，太小了无法捕捉有用信息，太多了全是没出现的样本，如何改进的问题
3： 训练数据取多少？n设置为多少？
4：如何利用信息？ （中文，） 贝叶斯网络
5：如果条件是独一无二的，那么答案一定是正确的， 但是完全没有泛化能力，因为后面的条件一顶不同
6：完全拼写正确的准确度是多少？ 不是距离，而是完全拼写正确的比例
7：音标如何辅助预测？
"""


import os
import random
from typing import List
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import ConditionalFreqDist, MLEProbDist
import Levenshtein
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------read vocabulary data---------------------------------------------------------------
CURRENT_PATH = os.getcwd()  # get the current path
# get the vocab data path
_vocabulary_absolute_path = os.path.join(CURRENT_PATH, 'vocabulary_books', 'CET4', 'Vocab.json')
# initialize vocabulary instance
_vocab_instance = ReadVocabBook(vocab_book_path=_vocabulary_absolute_path,
                                vocab_book_name='CET4',
                                chinese_setting=True,
                                phonetic_setting=True,
                                POS_setting=True,
                                english_setting=True)

# read vocab data [[condition, english]......] [['j ɔ r', 'y o u r'], ['j ɝ s ɛ l f', 'y o u r s e l f']]
_vocab_data: List[List[str]] = _vocab_instance.read_vocab_book()
print(_vocab_data)
# get the longest length and shortest length
word_length_list = []
for task in _vocab_data:
    word_length_list.append(len(''.join(task[1].split())))
# print(f' the maximum word length is: {max(word_length_list)}, and the shortest word length is: {min(word_length_list)}',)
LONGEST_WORD_LENGTH = max(word_length_list)  # 14

random.shuffle(_vocab_data)  # shuffle the vocabulary book


# ----------------------------------------------train model-------------------------------------------------------------

# nltk.download('punkt') # down the necessary file for tokenize


def generate_ngrams(tokens, n):
    """ generate n-grams with start, end and pad symbol"""
    start = ['<bos>']  # start can be information
    end = ['<eos>']
    pad = ['<pad>']
    tokens = start + tokens + end
    padding_part = pad * (LONGEST_WORD_LENGTH + 2 - len(tokens))  # padding based on the longest word length
    padding_token = tokens + padding_part
    # print(f' the token list after padding: {padding_token}')
    return list(ngrams(padding_token, n))


def construct_dataset(vocab_data, split_ratio, n_length, condition_length):
    """ generate n_grams and conditions """

    training_data = vocab_data[: int(len(vocab_data) * split_ratio)]
    testing_data = vocab_data[int(len(vocab_data) * split_ratio):]

    training_tokens = [word_tokenize(data[1]) for data in training_data]
    testing_tokens = [word_tokenize(data[1]) for data in testing_data]

    training_seq_ngrams = [generate_ngrams(seq_token, n_length) for seq_token in training_tokens]
    training_n_grams = [ngram for seq_ngrams in training_seq_ngrams for ngram in seq_ngrams]

    testing_seq_ngrams = [generate_ngrams(seq_token, n_length) for seq_token in testing_tokens]
    testing_n_grams = [ngram for seq_ngrams in testing_seq_ngrams for ngram in seq_ngrams]

    training_condition_grams_with_bos = []
    testing_condition_grams_with_bos = []

    # read the condition with '<bos>'
    for grams in training_n_grams:
        if list(grams)[0] == '<bos>':
            training_condition_grams_with_bos.append(list(grams)[:condition_length])

    for grams in testing_n_grams:
        if list(grams)[0] == '<bos>':
            testing_condition_grams_with_bos.append(list(grams)[:condition_length])
    # print("condition grams from '<bos>':", condition_grams)
    # print(f' the length of conditional grams with <bos>: {len(condition_grams_with_bos)}')

    return training_data, testing_data, training_n_grams, training_condition_grams_with_bos, testing_n_grams, testing_condition_grams_with_bos


def calculate_condition_feq(n_grams, condition_length):
    """ train model: calculate frequency matrix """
    # calculate frequency distribution, (condition, event)
    cond_freq_dist = ConditionalFreqDist(
        (tuple(ngram[:condition_length]), ngram[condition_length]) for ngram in n_grams)

    prob_dist = MLEProbDist(cond_freq_dist)
    # print(prob_dist)
    return cond_freq_dist


def generate_spelling(model, condition_grams_with_bos, condition_length):
    """ generate spelling based on trained model"""
    generated_answer = []  # the generated content via probability matrix
    word_index = 0
    for con_grams in condition_grams_with_bos:
        for _ in range(LONGEST_WORD_LENGTH + 1 - condition_length):
            try:
                condition = tuple(con_grams[-condition_length:])
                next_word = model[condition].max()  # select the maximum probability
                con_grams.append(next_word)
            except ValueError as e:
                print(f"Error: {e}")
                print(f"No samples for '{condition}'. Add some samples first.")

        # only read the legal letters
        pure_con_grams = [letter for letter in con_grams if letter not in ['<bos>', '<eos>', '<pad>']]
        generated_answer.append(" ".join(pure_con_grams))
        word_index += 1
        # print("Generated Text:", generated_answer)
    return generated_answer


def evaluation_model(data, generated_answer):
    """ evaluation according to accuracy and completeness"""
    accuracy_list = []
    completeness_list = []
    perfect_spelling_counts = 0
    for index in range(len(data)):

        stu_spelling = ''.join(generated_answer[index].split(' '))
        correct_answer = ''.join(data[index][1].split(' '))
        # calculate Levenshtein accuracy and completeness
        word_accuracy = round(Levenshtein.ratio(stu_spelling, correct_answer), 2)
        word_completeness = round(1 - Levenshtein.distance(stu_spelling, correct_answer) / word_length_list[index], 2)
        accuracy_list.append(word_accuracy)
        completeness_list.append(word_completeness)
        # calculate perfect spelling accuracy
        if stu_spelling == correct_answer:
            perfect_spelling_counts += 1

    # 在这判断perfect spelling 的比例
    n_avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    n_avg_completeness = sum(completeness_list) / len(completeness_list)
    perfect_spelling_ratio = perfect_spelling_counts / len(accuracy_list)
    return n_avg_accuracy, n_avg_completeness, perfect_spelling_ratio


def draw_n_plots(vocab_data, split_ratio):
    """explore the relationship between n and vocabulary size"""
    N_LENGTH = [n_length for n_length in range(2, LONGEST_WORD_LENGTH + 2, 1)]  # test from 2 to 15 [2-15]
    # print(_n_length)
    CONDITION_LENGTH = [con_length - 1 for con_length in N_LENGTH]
    # print(_condition_length)
    n_accuracy = []
    n_completeness = []
    n_perfect_ratio = []
    for n_index in range(len(N_LENGTH)):
        train_data, test_data, train_grams, train_condition, \
            test_grams, test_condition = construct_dataset(vocab_data,
                                                           split_ratio,
                                                           N_LENGTH[n_index],
                                                           CONDITION_LENGTH[n_index])

        freq_model = calculate_condition_feq(train_grams, CONDITION_LENGTH[n_index])
        # generated_answer = generate_spelling(freq_model, test_condition, CONDITION_LENGTH[n_index])  # change condition
        generated_answer = generate_spelling(freq_model, test_condition, CONDITION_LENGTH[n_index])  # change condition
        n_avg_accuracy, n_avg_completeness, perfect_spelling_ratio = evaluation_model(train_data, generated_answer)
        n_accuracy.append(n_avg_accuracy)
        n_completeness.append(n_avg_completeness)
        n_perfect_ratio.append(perfect_spelling_ratio)
    # plt.plot(_n_length, n_accuracy, label='n_accuracy', color='blue')
    # plt.plot(_n_length, n_completeness, label='n_completeness', color='red')
    # plt.xlabel(f'data size {len(generated_answer)} in different n')
    # plt.ylabel('evaluation metrics')
    # plt.xticks(_n_length)  # 设置为你的 x 数据
    # plt.legend()
    # plt.show()
    return n_accuracy, n_completeness, n_perfect_ratio


if __name__ == '__main__':
    _split_ratio = [size / len(_vocab_data) for size in range(50, 1050, 100)]
    _n_length = [n_length for n_length in range(2, LONGEST_WORD_LENGTH + 2, 1)]  # test from 2 to 14 [2-14]
    _condition_length = [con_length - 1 for con_length in _n_length]
    accuracy = []
    completeness = []
    perfect_ratio = []
    for ratio in _split_ratio:
        avg_accuracy, avg_completeness, n_perfect_ratio = draw_n_plots(_vocab_data, ratio)
        print(f'data size:{ratio}, avg_accuracy:{avg_accuracy}, avg_completeness:{avg_completeness}, perfect_ratio:{n_perfect_ratio}')
        accuracy.append(avg_accuracy)
        completeness.append(avg_completeness)
        perfect_ratio.append(n_perfect_ratio)
    acc_df = pd.DataFrame(accuracy, columns=_n_length, index=_split_ratio)
    com_df = pd.DataFrame(completeness, columns=_n_length, index=_split_ratio)
    per_df = pd.DataFrame(perfect_ratio, columns=_n_length, index=_split_ratio)
    # 将 DataFrame 保存为 CSV 文件
    acc_df.to_excel('test_data/pure_ngrams/acc_data.xls')
    com_df.to_excel('test_data/pure_ngrams/com_data.xls')
    per_df.to_excel('test_data/pure_ngrams/perfect_spelling.xls')
