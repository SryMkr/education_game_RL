"""
This file is to achieve dictionary student. it like the student know some words.
"""
import random
import Levenshtein
import os
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook
from typing import List, Dict, Tuple
from itertools import chain
from collections import Counter
import pandas as pd
import string
from ngrams import laplace_smoothing, good_turning

CURRENT_PATH = os.getcwd()  # get the current path
VOCAB_PATH: str = os.path.join(CURRENT_PATH, 'vocabulary_books', 'CET4', 'newVocab.json')  # get the vocab data path
corpus_instance = ReadVocabBook(vocab_book_path=VOCAB_PATH,
                                vocab_book_name='CET4',
                                chinese_setting=False,
                                phonetic_setting=True,
                                POS_setting=False,
                                english_setting=True)
original_corpus = corpus_instance.read_vocab_book()

random.shuffle(original_corpus)  # shuffle the vocabulary book
training_corpus = original_corpus[: int(len(original_corpus) * 0.8)]  # get the training data [phonemes, word]
testing_corpus = original_corpus[int(len(original_corpus) * 0.8):]  # get the testing data
# print(testing_corpus)


class DictionaryStudent:
    """achieve dictionary student"""

    def __init__(self, train_corpus: List[List[str]], test_corpus: List[List[str]]):
        """accept corpus"""
        self.train_corpus: Dict[str, str] = {row[0]: row[1].replace(' ', '') for row in train_corpus}
        self.test_corpus: Dict[str, str] = {row[0]: row[1].replace(' ', '') for row in test_corpus}
        self.student_answer_pair = []
        self.accuracy = []
        self.completeness = []
        self.perfect = []
        self.avg_accuracy = 0.0
        self.avg_completeness = 0.0
        self.avg_perfect = 0.0

    def generate_answer(self) -> None:
        """generate spelling based on phonemes"""
        for phonemes, answer in self.test_corpus.items():
            if phonemes in self.train_corpus.keys():
                self.student_answer_pair.append([self.train_corpus[phonemes], answer])
            else:
                self.student_answer_pair.append(['', answer])

    def evaluation(self) -> Tuple[float, float, float]:
        """evaluate the model in terms of accuracy, completeness, perfect"""
        for stu_answer, correct_answer in self.student_answer_pair:
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
    def __init__(self, train_corpus: List[List[str]], test_corpus: List[List[str]], n):
        self.train_corpus: List[str] = [''.join(word_pair[1].split()) for word_pair in train_corpus]
        self.test_corpus: List[str] = [''.join(word_pair[1].split()) for word_pair in test_corpus]
        self.n = n
        self.condition_len = n - 1
        self.grams: List[List[str]] = []
        self.conditions: List[List[str]] = []
        self.vocab_size: int = 0
        self.freq_df = None
        self.prob_df = None
        self.student_answer_pair = []
        self.accuracy = []
        self.completeness = []
        self.perfect = []
        self.avg_accuracy = 0.0
        self.avg_completeness = 0.0
        self.avg_perfect = 0.0

    def generate_gram(self) -> None:
        for word in self.train_corpus:
            n_grams = [word[i:i + self.n] for i in range(len(word) - self.n + 1)]  # generate grams
            n_conditions = [word[i:i + self.condition_len] for i in
                            range(len(word) - self.n + 1)]  # generate conditions
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
            freq_df.loc[ngrams[:-1], ngrams[-1]] = counts
        self.freq_df = freq_df.fillna(0)
        self.prob_df = freq_df.div(conditions_counts, axis=0)
        self.prob_df = self.prob_df.fillna(0)  # frequency table
        if smoothing_method == 'laplace':
            self.freq_df, self.prob_df = laplace_smoothing(self.freq_df, conditions_counts, self.vocab_size)
        if smoothing_method == 'gt':
            self.freq_df, self.prob_df = good_turning(self.freq_df, conditions_counts)

    def generate_answer(self) -> None:
        """predict result"""
        for word in self.test_corpus:
            stu_answer = word[:self.condition_len]  # get the task condition
            task_length = len(word)
            # control the length of prediction
            for _ in range(task_length - self.condition_len):
                condition_window = stu_answer[-self.condition_len:]
                # 如果condition_window,不在词表中,随机挑选一个字母，不重新训练模型！！！
                if condition_window not in self.prob_df.index:
                    next_letter = random.choice(string.ascii_lowercase)
                else:
                    next_letter = self.prob_df.loc[condition_window].idxmax()  # select the maximum probability
                stu_answer += next_letter
            self.student_answer_pair.append([stu_answer, word])

    def evaluation(self) -> Tuple[float, float, float]:
        for stu_answer, correct_answer in self.student_answer_pair:
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


class PhoLetStudent:
    def __init__(self, train_corpus: List[List[str]], test_corpus: List[List[str]]):
        self.train_corpus = [[item.split() for item in sublist] for sublist in train_corpus]
        self.test_corpus = [[item.split() for item in sublist] for sublist in test_corpus]
        self.letters: List[str] = []
        self.phonemes: List[str] = []
        self.phoneme_letter_prob = {}
        self.phoneme_letter_df = pd.DataFrame()
        self.student_answer_pair = []
        self.accuracy = []
        self.completeness = []
        self.perfect = []
        self.avg_accuracy = 0.0
        self.avg_completeness = 0.0
        self.avg_perfect = 0.0

    def construct_vocab(self) -> None:
        letters = []
        phonemes = []
        for sp in self.train_corpus:
            for fw in sp[0]:  # phoneme
                phonemes.append(fw)
            for ew in sp[1]:  # letters
                letters.append(ew)
        # covert into lower letter, and omit the duplicated word
        self.letters = sorted(list(set(letters)), key=lambda s: s.lower())  # 26
        self.phonemes = sorted(list(set(phonemes)), key=lambda s: s.lower())  # 39

    def initial_prob(self) -> None:
        """ initialize all prob"""
        init_prob = 1.0 / len(self.letters)
        for fw in self.phonemes:
            for ew in self.letters:
                if fw not in self.phoneme_letter_prob:
                    self.phoneme_letter_prob[fw] = {}  # create the dictionary for each phoneme
                self.phoneme_letter_prob[fw][ew] = init_prob

    def train_model(self, num_epochs=20):
        """ train the model"""
        s_total = {}
        for epoch in range(num_epochs):
            phoneme_letter_counts = {}
            total = {}
            # initialize the counts of fw-ew pair
            for fw in self.phonemes:
                total[fw] = 0.0
                for ew in self.letters:
                    if fw not in phoneme_letter_counts:
                        phoneme_letter_counts[fw] = {}
                    phoneme_letter_counts[fw][ew] = 0.0
            # print('fw-ew pairs: ', phoneme_letter_counts)

            for sp in self.train_corpus:
                # 对于每一个corpus，将所有的外文单词对应到每一个英文单词上面的概率值相加，相当于不同的对齐方式
                for ew in sp[1]:  # 循环英文字母
                    s_total[ew] = 0.0
                    for fw in sp[0]:
                        s_total[ew] += self.phoneme_letter_prob[fw][ew]

                for ew in sp[1]:
                    # 将每一种对齐关系中的概率除于总的概率（所有对齐关系）近似于求出每一种对齐方式的概率，相加就是对应关系的期望数量
                    for fw in sp[0]:
                        phoneme_letter_counts[fw][ew] += self.phoneme_letter_prob[fw][ew] / s_total[
                            ew]  # 对于任何一种对齐方式  求出其期望数量
                        total[fw] += self.phoneme_letter_prob[fw][ew] / s_total[ew]  # 求出不同外文单词的概率

            # normalization
            for fw in self.phonemes:
                for ew in self.letters:
                    self.phoneme_letter_prob[fw][ew] = phoneme_letter_counts[fw][ew] / total[fw]
        self.phoneme_letter_df = pd.DataFrame.from_dict(self.phoneme_letter_prob, orient='index')

    def generate_answer(self):
        """ generate answer based on the given phonemes"""
        for phonemes, answer in self.test_corpus:
            spelling = ''
            answer = ''.join(answer)
            for phoneme in phonemes:
                letter = self.phoneme_letter_df.loc[phoneme].idxmax()
                spelling = spelling + letter
            self.student_answer_pair.append([spelling, answer])

    def evaluation(self) -> Tuple[float, float, float]:
        for stu_answer, correct_answer in self.student_answer_pair:
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


def connect_adjacent_index(letters, phonemes):
    # 初始化合并后的列表
    merged_list = []
    i = 0
    while i < len(letters):
        # 将当前字母添加到合并后的列表
        current_letter = letters[i]
        # 检查下一个字母是否与当前字母相同，如果相同则合并
        while i + 1 < len(phonemes) and phonemes[i] == phonemes[i + 1]:
            i += 1
            current_letter += letters[i]
        merged_list.append(current_letter)
        i += 1
    return ' '.join(merged_list)

def traceback(result_df):
    phonemes_seq = []
    letters_seq = []
    p_index, l_index = result_df.shape[0] - 1, result_df.shape[1] - 1
    while p_index >= 0 and l_index >= 0:
        current_score = result_df.iloc[p_index, l_index]
        diagonal_score = result_df.iloc[p_index - 1, l_index - 1]
        left_score = result_df.iloc[p_index, l_index - 1]
        # 现在该比较大小了，全部按照概率来排有问题，完全按照最大概率会出现有的音标没有用到的情况
        # 解决方案就是一种一种的试，直到试出符合要求的结果
        max_score, direction = max((diagonal_score, 'diagonal'), (left_score, 'left'))
        current_score += max_score
        phonemes_seq.append(result_df.index[p_index])
        letters_seq.append(result_df.columns[l_index])
        if direction == 'diagonal':
            p_index -= 1
            l_index -= 1
        else:  # move == 'left'
            l_index -= 1
    phonemes_seq.reverse()
    letters_seq.reverse()
    # 如果长度不相等还需要继续检索，需要设置回退机制
    return phonemes_seq, letters_seq


class PhoLetsStudent:
    """achieve phoneme letters combo student"""

    def __init__(self, phoneme_letter_prob, train_corpus: List[List[str]], test_corpus: List[List[str]]):
        self.phoneme_letter_prob = phoneme_letter_prob
        self.test_corpus = test_corpus
        self.train_corpus = [[item.split() for item in sublist] for sublist in train_corpus]
        self.phoneme_letters = {}
        self.letters = []  # store all combo
        self.new_train_corpus = []
        self.avg_accuracy = 0.0
        self.avg_completeness = 0.0
        self.avg_perfect = 0.0

    def find_combo(self):
        """fine the possible combo,保证音标的长度和最后字母区分的长度一样，完全按照概率肯定行不通"""
        for phonemes, letters in self.train_corpus:
            if len(phonemes) == len(letters) or len(phonemes) == 0:
                continue
            else:
                result_df = self.phoneme_letter_prob.loc[phonemes, letters].round(4)
                # print(result_df)
                max_phonemes, max_letters = traceback(result_df)
                # print(max_phonemes, max_letters)
                word = connect_adjacent_index(max_letters, max_phonemes)  # [['p aʊ ɝ', 'p o w e r']]
                # print(word)
                # 将word分割，如果分割之后的字母长度有大于三的则合并失败
                result = any(len(element) > 3 for element in word.split())
                if result:
                    self.new_train_corpus.append([' '.join(phonemes), ' '.join(letters)])
                else:
                    self.new_train_corpus.append([' '.join(phonemes), word])
        # print(self.new_train_corpus)

    def evaluation(self) -> Tuple[float, float, float]:
        """formulate new model"""
        phoLet_student_1 = PhoLetStudent(self.new_train_corpus, self.test_corpus)
        phoLet_student_1.construct_vocab()
        phoLet_student_1.initial_prob()
        phoLet_student_1.train_model()
        phoLet_student_1.phoneme_letter_df.round(4).to_excel('phoneme_combo_df.xls')
        phoLet_student_1.generate_answer()
        self.avg_accuracy, self.avg_completeness, self.avg_perfect = phoLet_student.evaluation()
        return self.avg_accuracy, self.avg_completeness, self.avg_perfect


if __name__ == '__main__':
    # dictionary student
    dic_student = DictionaryStudent(training_corpus, testing_corpus)
    dic_student.generate_answer()
    dic_accuracy, dic_completeness, dic_perfect = dic_student.evaluation()
    print(f'dictionary students accuracy is: {dic_accuracy}, completeness is: {dic_completeness}, perfect is: {dic_perfect}')
    # bigram student
    bigram_student = NgramStudent(training_corpus, testing_corpus, 2)
    bigram_student.generate_gram()
    bigram_student.train_model('laplace')
    bigram_student.generate_answer()
    Big_accuracy, Big_completeness, Big_perfect = bigram_student.evaluation()
    print(f'bigram students accuracy is: {Big_accuracy}, completeness is: {Big_completeness}, perfect is: {Big_perfect}')
    # trigram student
    trigram_student = NgramStudent(training_corpus, testing_corpus, 3)
    trigram_student.generate_gram()
    trigram_student.train_model('laplace')
    trigram_student.generate_answer()
    trig_accuracy, trig_completeness, trig_perfect = trigram_student.evaluation()
    print(
        f'trigram students accuracy is: {trig_accuracy}, completeness is: {trig_completeness}, perfect is: {trig_perfect}')

    # phoneme letter pair student
    phoLet_student = PhoLetStudent(training_corpus, testing_corpus)
    phoLet_student.construct_vocab()
    phoLet_student.initial_prob()
    phoLet_student.train_model()
    phoLet_student.generate_answer()
    phoLet_accuracy, phoLet_completeness, phoLet_perfect = phoLet_student.evaluation()
    print(f'phoLet students accuracy is: {phoLet_accuracy}, completeness is: {phoLet_completeness}, '
          f'perfect is: {phoLet_perfect}')
    # phoneme letters combo student
    pholets_student = PhoLetsStudent(phoLet_student.phoneme_letter_df, training_corpus, testing_corpus)
    # pholets_student = PhoLetsStudent(phoLet_student.phoneme_letter_df, aaa_corpus, testing_corpus)
    pholets_student.find_combo()
    phoLets_accuracy, phoLets_completeness, phoLets_perfect = pholets_student.evaluation()
    print(f'phoLets students accuracy is: {phoLet_accuracy}, completeness is: {phoLet_completeness}, '
          f'perfect is: {phoLet_perfect}')
