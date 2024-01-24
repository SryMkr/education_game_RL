"""ngram with position
The maximum of length is 10/9
This model is used to simulate excellent student agent
"""

import random
import Levenshtein
import os

from Spelling_Framework.agents.Dictionary_Student import PhoLetStudent
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook
from typing import List, Tuple
import pandas as pd
import string


CURRENT_PATH = os.getcwd()  # get the current path
VOCAB_PATH: str = os.path.join(CURRENT_PATH, 'vocabulary_books', 'CET4', 'newVocab.json')  # get the vocab data path
STU_MEMORY_PATH = os.path.join(CURRENT_PATH, 'stu_memory.xlsx')
corpus_instance = ReadVocabBook(vocab_book_path=VOCAB_PATH,
                                vocab_book_name='CET4',
                                chinese_setting=False,
                                phonetic_setting=True,
                                POS_setting=False,
                                english_setting=True)
original_corpus = corpus_instance.read_vocab_book()
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

# 在这直接构造DF，横坐标和纵坐标都弄上9个，然后初始化
for i in range(10):
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

def add_position(corpus):
    """add position for each corpus"""
    corpus_with_position = []
    for pair in corpus:
        phonemes_position = ''
        letters_position = ''
        pair_position = []
        phonemes_list = pair[0].split(' ')
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



t = add_position(original_corpus)



pos_training_corpus = add_position(training_corpus)  # get the training data [phonemes, word]
pos_testing_corpus = add_position(testing_corpus)  # get the testing data





class PositionPhoLetStudent:
    def __init__(self, train_corpus: List[List[str]], test_corpus: List[List[str]], initial_prob, p_index, l_columns):
        self.train_corpus = [[item.split() for item in sublist] for sublist in train_corpus]
        self.test_corpus = [[item.split() for item in sublist] for sublist in test_corpus]
        self.letters: List[str] = l_columns
        self.phonemes: List[str] = p_index
        self.phoneme_letter_prob = initial_prob
        self.phoneme_letter_df = pd.DataFrame()
        self.student_answer_pair = []
        self.accuracy = []
        self.completeness = []
        self.perfect = []
        self.avg_accuracy = 0.0
        self.avg_completeness = 0.0
        self.avg_perfect = 0.0

    def train_model(self, num_epochs=10):
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
        self.phoneme_letter_df.replace(0, 0.0001, inplace=True)  # replace zero with constant

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
                    letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                    letter = letters_prob.idxmax()
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


if __name__ == "__main__":
    # phoneme letter pair student
    phoLet_student = PhoLetStudent(training_corpus, original_corpus)
    phoLet_student.construct_vocab()
    phoLet_student.initial_prob()
    phoLet_student.train_model()
    phoLet_student.generate_answer()
    phoLet_accuracy, phoLet_completeness, phoLet_perfect = phoLet_student.evaluation()
    print(f'phoLet students accuracy is: {phoLet_accuracy}, completeness is: {phoLet_completeness}, '
          f'perfect is: {phoLet_perfect}')

    # 训练音标和字母之间的关系
    # phoneme letter pair student
    position_phoLet_student = PositionPhoLetStudent(pos_training_corpus, t, phoneme_letter_prob, df_index, df_column)
    position_phoLet_student.train_model()
    # position_phoLet_student.phoneme_letter_df.to_excel(STU_MEMORY_PATH, engine='openpyxl')
    position_phoLet_student.generate_answer()
    position_phoLet_accuracy, position_phoLet_completeness, position_phoLet_perfect = position_phoLet_student.evaluation()
    print(f'position phoLet students accuracy is: {position_phoLet_accuracy}, completeness is: {position_phoLet_completeness}, '
          f'perfect is: {position_phoLet_perfect}')
