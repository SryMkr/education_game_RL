"""ngram with position
The maximum of length is 14/14
词库中尽然有音标为空者
根据音标拼写单词，提供答案长度，并且一个个拼写。看看准确度和完整度
使用添加了位置的根据音标来拼写单词
"""

import random
import Levenshtein
import os

from Spelling_Framework.agents.Dictionary_Student import PhoLetStudent
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook
from typing import List, Dict, Tuple
from itertools import chain
from collections import Counter
import pandas as pd
import string
from ngrams import laplace_smoothing, good_turning


CURRENT_PATH = os.getcwd()  # get the current path
VOCAB_PATH: str = os.path.join(CURRENT_PATH, 'vocabulary_books', 'CET4', 'Vocab.json')  # get the vocab data path
corpus_instance = ReadVocabBook(vocab_book_path=VOCAB_PATH,
                                vocab_book_name='CET4',
                                chinese_setting=False,
                                phonetic_setting=True,
                                POS_setting=False,
                                english_setting=True)
original_corpus = corpus_instance.read_vocab_book()
# original_corpus = [['p ɑ p j ʌ l eɪ ʃ ʌ n', 'p o p u l a t i o n'], ['n aɪ n t i n', 'n i n e t e e n']]
random.shuffle(original_corpus)  # [['p ɑ p j ʌ l eɪ ʃ ʌ n', 'p o p u l a t i o n'], ['n aɪ n t i n', 'n i n e t e e n']
training_corpus = original_corpus[:int(len(original_corpus) * 0.8)]  # get the training data [phonemes, word]
testing_corpus = original_corpus[int(len(original_corpus) * 0.8):]  # get the testing data


corpus_1 = [[item.split() for item in sublist] for sublist in original_corpus]
letters_1 = []
phonemes_1 = []
for sp in corpus_1:
    for fw in sp[0]:  # phoneme
        phonemes_1.append(fw)
    for ew in sp[1]:  # letters
        letters_1.append(ew)
# covert into lower letter, and omit the duplicated word
letters_set = sorted(list(set(letters_1)), key=lambda s: s.lower())  # 26
phonemes_set = sorted(list(set(phonemes_1)), key=lambda s: s.lower())  # 39
# print(letters_set)
# print(phonemes_set)

df_index = []
df_column = []

# 在这直接构造DF，横坐标和纵坐标都弄上14个，然后初始化
for i in range(14):
    l_p = ''
    p_p = ''
    for l_i in letters_set:
        l_p = l_i + '_' + str(i)
        df_column.append(l_p)
    for p_i in phonemes_set:
        p_p = p_i + '_' + str(i)
        df_index.append(p_p)

""" initialize all prob, all possible included"""
init_prob = 1/len(df_column)
phoneme_letter_prob = pd.DataFrame(init_prob, index=df_index, columns=df_column)
# print(phoneme_letter_prob)
# phoneme_letter_prob.to_excel('output.xlsx', engine='openpyxl')


def add_position(corpus):
    """add position for each corpus"""
    corpus_with_position = []
    for pair in corpus:
        phonemes_position = ''
        letters_position = ''
        pair_position = []
        phonemes_list = pair[0].split(' ')
        if len(pair[0]) == 0 or len(pair[1]) == 0:  # 去除为空的情况
            continue
        else:
            for index, phoneme in enumerate(phonemes_list):
                phoneme_index = phoneme + '_' + str(index)
                phonemes_position = phonemes_position + phoneme_index + ' '
            letters_list = pair[1].split(' ')

            for index, letter in enumerate(letters_list):
                letter_index = letter + '_' + str(index)
                letters_position = letters_position + letter_index + ' '
            pair_position.append(phonemes_position.strip())
            pair_position.append(letters_position.strip())
            corpus_with_position.append(pair_position)
    return corpus_with_position


pos_training_corpus = add_position(training_corpus)  # get the training data [phonemes, word]
pos_testing_corpus = add_position(testing_corpus)  # get the testing data

class PositionPhoLetStudent:
    def __init__(self, train_corpus: List[List[str]], test_corpus: List[List[str]], initial_prob, p_index, l_columns, bigram_prob):
        self.train_corpus = [[item.split() for item in sublist] for sublist in train_corpus]
        self.test_corpus = [[item.split() for item in sublist] for sublist in test_corpus]
        self.letters: List[str] = l_columns
        self.phonemes: List[str] = p_index
        self.phoneme_letter_prob = initial_prob
        self.bigram_prob = bigram_prob
        self.phoneme_letter_df = pd.DataFrame()
        self.student_answer_pair = []
        self.accuracy = []
        self.completeness = []
        self.perfect = []
        self.avg_accuracy = 0.0
        self.avg_completeness = 0.0
        self.avg_perfect = 0.0

    def train_model(self, num_epochs=6):
        """ train the model"""
        s_total = {}
        for epoch in range(num_epochs):
            phoneme_letter_counts = {}
            total = {}
            # initialize the counts of fw-ew pair
            for fw in self.phonemes:
                total[fw] = 0.00001
                for ew in self.letters:
                    if fw not in phoneme_letter_counts:
                        phoneme_letter_counts[fw] = {}
                    phoneme_letter_counts[fw][ew] = 0
            # print('fw-ew pairs: ', phoneme_letter_counts)

            for sp in self.train_corpus:
                # 对于每一个corpus，将所有的外文单词对应到每一个英文单词上面的概率值相加，相当于不同的对齐方式
                for ew in sp[1]:  # 循环英文字母['n aɪ n t i n', 'n i n e t e e n']
                    s_total[ew] = 0.0
                    for fw in sp[0]:
                        s_total[ew] += self.phoneme_letter_prob.loc[fw, ew]

                for ew in sp[1]:
                    # 将每一种对齐关系中的概率除于总的概率（所有对齐关系）近似于求出每一种对齐方式的概率，相加就是对应关系的期望数量
                    for fw in sp[0]:
                        # 对于任何一种对齐方式  求出其期望数量
                        phoneme_letter_counts[fw][ew] += self.phoneme_letter_prob.loc[fw, ew] / s_total[ew]
                        total[fw] += self.phoneme_letter_prob.loc[fw, ew] / s_total[ew]  # 求出不同外文单词的概率

            # normalization
            for fw in self.phonemes:
                for ew in self.letters:
                    self.phoneme_letter_df.loc[fw, ew] = phoneme_letter_counts[fw][ew] / total[fw]

        self.phoneme_letter_df.replace(0, 0.0001)  # replace zero with constant
        # self.phoneme_letter_df = self.phoneme_letter_prob.div(self.phoneme_letter_prob.sum(axis=1), axis=0)

    def generate_answer(self):
        """ generate answer based on the given phonemes,而且我要知道答案的长度，然后根据所有的音标对每一个位置选择最大值"""
        for phonemes, answer in self.test_corpus:
            spelling = []
            answer_length = len(answer)
            alphabet = string.ascii_lowercase
            for i in range(answer_length):
                # 将26个字母和位置结合起来，组成列索引
                if i == 0:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.phoneme_letter_df.loc[phonemes[0], result_columns]
                    letter = possible_results.idxmax()
                else:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.phoneme_letter_df.loc[phonemes, result_columns]
                    letters_prob = possible_results.sum(axis=0).to_dict()  # 每一列相加,取概率最大值
                    if spelling[i - 1] not in self.bigram_prob.index:
                        self.bigram_prob.loc[spelling[i - 1], :] = 0.00001
                    bigram_prob = self.bigram_prob.loc[spelling[i - 1]].to_dict()
                    # 将pd变为字典，key是字母，value是概率
                    products = {key: bigram_prob[key] + letters_prob[key] for key in letters_prob if key in bigram_prob}
                    # 让两个概率相乘取大者
                    letter = max(products, key=products.get)
                spelling.append(letter)
            self.student_answer_pair.append([spelling, answer])

    def evaluation(self) -> Tuple[float, float, float]:
        for stu_answer, correct_answer in self.student_answer_pair:
            stu_answer = ''.join([i.split('_')[0] for i in stu_answer])
            correct_answer = ''.join([i.split('_')[0] for i in correct_answer])
            word_accuracy = round(Levenshtein.ratio(correct_answer, stu_answer), 2)
            word_completeness = round(1 - Levenshtein.distance(correct_answer, stu_answer) / len(correct_answer), 2)
            word_perfect = 0.0
            if stu_answer == correct_answer:
                word_perfect = 1.0
            self.accuracy.append(word_accuracy)
            self.completeness.append(word_completeness)
            self.perfect.append(word_perfect)
        self.avg_accuracy = sum(self.accuracy) / len(self.accuracy)
        self.avg_completeness = sum(self.completeness) / len(self.completeness)
        self.avg_perfect = sum(self.perfect) / len(self.perfect)
        return self.avg_accuracy, self.avg_completeness, self.avg_perfect

class NgramStudent:
    # ['r_0 ɪ_1 m_2 eɪ_3 n_4', 'r_0 e_1 m_2 a_3 i_4 n_5']
    def __init__(self, train_corpus: List[List[str]], test_corpus: List[List[str]], n):
        self.train_corpus = [word_pair[1].split(' ') for word_pair in train_corpus]
        self.test_corpus: List[str] = [''.join(word_pair[1].split()) for word_pair in test_corpus]
        self.n = n
        self.condition_len = n - 1
        self.grams = []
        self.conditions: List[List[str]] = []
        self.vocab_size: int = 0
        self.freq_df = None
        self.prob_df = None

    def generate_gram(self) -> None:
        # ['s_0', 'p_1', 'o_2', 'n_3', 't_4', 'a_5', 'n_6', 'e_7', 'o_8', 'u_9', 's_10']
        for word in self.train_corpus:
            n_grams = [tuple(word[i:i + self.n]) for i in range(len(word) - self.n + 1)]  # generate grams
            n_conditions = word[:-1]  # generate conditions
            self.grams.append(n_grams)
            self.conditions.append(n_conditions)
        self.grams = list(chain(*self.grams))
        self.conditions = list(chain(*self.conditions))

    def train_model(self, smoothing_method='empty') -> None:
        conditions_counts = dict(Counter(self.conditions))  # convert tuple into dictionary
        ngrams_counts = dict(Counter(self.grams))
        self.vocab_size = len(ngrams_counts.keys())  # for smoothing
        freq_df = pd.DataFrame()
        for ngrams, counts in ngrams_counts.items():
            freq_df.loc[ngrams[0], ngrams[1]] = counts
        self.freq_df = freq_df.fillna(0)
        self.prob_df = freq_df.div(conditions_counts, axis=0)
        self.prob_df = self.prob_df.fillna(0)  # frequency table

        if smoothing_method == 'laplace':
            self.freq_df, self.prob_df = laplace_smoothing(self.freq_df, conditions_counts, self.vocab_size)
        if smoothing_method == 'gt':
            self.freq_df, self.prob_df = good_turning(self.freq_df, conditions_counts)


if __name__ == "__main__":
    # phoneme letter pair student
    phoLet_student = PhoLetStudent(training_corpus, testing_corpus)
    phoLet_student.construct_vocab()
    phoLet_student.initial_prob()
    phoLet_student.train_model()
    phoLet_student.generate_answer()
    phoLet_accuracy, phoLet_completeness, phoLet_perfect = phoLet_student.evaluation()
    print(f'phoLet students accuracy is: {phoLet_accuracy}, completeness is: {phoLet_completeness}, '
          f'perfect is: {phoLet_perfect}')

    # 训练unigram, bigram，trigram,尝试使用n_grams来纠正纯概率拼写
    # bigram student
    bigram_student = NgramStudent(pos_training_corpus, pos_testing_corpus, 2)
    bigram_student.generate_gram()
    bigram_student.train_model('laplace')

    # 训练音标和字母之间的关系
    # phoneme letter pair student
    position_phoLet_student = PositionPhoLetStudent(pos_training_corpus, pos_testing_corpus, phoneme_letter_prob, df_index, df_column,
                                                    bigram_student.prob_df)
    position_phoLet_student.train_model()
    # phoLet_student.phoneme_letter_df.to_excel('output.xlsx', engine='openpyxl')
    position_phoLet_student.generate_answer()
    position_phoLet_accuracy, position_phoLet_completeness, position_phoLet_perfect = position_phoLet_student.evaluation()
    print(f'position phoLet students accuracy is: {position_phoLet_accuracy}, completeness is: {position_phoLet_completeness}, '
          f'perfect is: {position_phoLet_perfect}')
