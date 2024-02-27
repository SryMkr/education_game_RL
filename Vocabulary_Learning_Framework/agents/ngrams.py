"""
this code is to simulate perfect students by using bigrams and trigrams

将四种方式都实现拼写，看看最后的结果怎么样
检查一下代码的执行结果是不是符合自己的设计，然后开始搞
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

CURRENT_PATH = os.getcwd()  # get the current path
VOCAB_PATH: str = os.path.join(CURRENT_PATH, 'vocabulary_books', 'CET4', 'newVocab.json')  # get the vocab data path

_phoneme_letter_abs_path = os.path.join(CURRENT_PATH,
                                        'test_data', 'phoneme_letter_pair', 'IBM_phoneme_letter_pair.xls')  # get the vocab data path
phoneme_letter_df = pd.read_excel(_phoneme_letter_abs_path, index_col=0, header=0)
# print(phoneme_letter_df)

def laplace_smoothing(freq_df, conditions_counts: Dict, vocabulary_size):
    freq_df = freq_df + 1
    conditions_counts = {key: value + vocabulary_size for key, value in conditions_counts.items()}
    prob_df = freq_df.div(conditions_counts, axis=0)
    # normalization
    row_sum = prob_df.sum(axis=1)
    prob_df = prob_df.div(row_sum, axis=0)
    return freq_df, prob_df


def good_turning(freq_df, conditions_counts: Dict):
    # 统计出现1次的ngram的数量
    count_zero = (freq_df == 0).sum().sum()
    count_one = (freq_df == 1).sum().sum()
    count_two = (freq_df == 2).sum().sum()
    count_three = (freq_df == 3).sum().sum()
    count_zero_star = count_one / count_zero + count_two / count_one + count_three / count_two
    freq_df[freq_df == 0] = count_zero_star
    prob_df = freq_df.div(conditions_counts, axis=0)
    # 归一化处理
    row_sum = prob_df.sum(axis=1)
    prob_df = prob_df.div(row_sum, axis=0)
    return freq_df, prob_df


class NGrams:
    """ achieve ngram algorithm"""
    def __init__(self, vocab_path, n):
        self.vocab_path = vocab_path  # the vocabulary path
        self.corpus: List[str] = []
        self.grams: List[List[str]] = []
        self.conditions: List[List[str]] = []
        self.vocab_size: int = 0
        self.freq_df = None
        self.prob_df = None
        self.n = n
        self.condition_len = n - 1
        self.unigram_prob = {}

    def get_corpus(self) -> None:
        """get the word corpus"""
        corpus_instance = ReadVocabBook(vocab_book_path=self.vocab_path,
                                        vocab_book_name='CET4',
                                        chinese_setting=False,
                                        phonetic_setting=False,
                                        POS_setting=False,
                                        english_setting=True)
        original_corpus = corpus_instance.read_vocab_book()
        self.corpus: List[str] = [''.join(word_pair[1].split()) for word_pair in original_corpus]  # the corpus 3613

    def unigram(self):
        """achieve unigram
        ['discard', 'power', 'competent']
        """
        letters = [letter for word in self.corpus for letter in word]
        vocab_size = len(letters)
        letters_counts = dict(Counter(letters))
        self.unigram_prob = {key: values/vocab_size for key, values in letters_counts.items()}

    def generate_grams(self) -> None:
        """generate grams according to corpus"""
        for word in self.corpus:
            n_grams = [word[i:i + self.n] for i in range(len(word) - self.n + 1)]  # generate grams
            n_conditions = [word[i:i + self.condition_len] for i in range(len(word) - self.n + 1)]  # generate conditions
            self.grams.append(n_grams)
            self.conditions.append(n_conditions)
        self.grams = list(chain(*self.grams))
        self.conditions = list(chain(*self.conditions))

    def calculate_prob(self, smoothing_method='empty') -> None:
        """calculate the matrix"""
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

    def calculate_seq_prob(self, text: str) -> Tuple[str, float]:
        word_prob = 1
        for i in range(len(text) - self.n + 1):
            ngrams = [text[i: i + self.condition_len], text[i + self.condition_len]]
            letter_prob = self.prob_df.loc[ngrams[0], ngrams[1]]
            print(f'ngrams{ngrams} prob is {letter_prob}')
            word_prob *= letter_prob
        return text, word_prob

    def predict(self, word: str) -> Tuple[str, str]:
        """predict result"""
        condition = word[:self.condition_len]  # get the task condition
        task_length = len(word)
        # control the length of prediction
        for _ in range(task_length - self.condition_len):
            condition_window = condition[-self.condition_len:]
            # 如果condition_window,不在词表中,随机挑选一个字母，不重新训练模型！！！
            if condition_window not in self.prob_df.index:
                next_letter = random.choice(string.ascii_lowercase)
            else:
                next_letter = self.prob_df.loc[condition_window].idxmax()  # select the maximum probability
            condition += next_letter
        return condition, word

    def evaluation(self, stu_answer: str, correct_answer: str) -> Tuple[float, float, float]:
        """evaluate the model"""
        word_accuracy = round(Levenshtein.ratio(correct_answer, stu_answer), 2)
        word_completeness = round(1 - Levenshtein.distance(correct_answer, stu_answer) / len(correct_answer), 2)
        word_perfect = 0.0
        if stu_answer == correct_answer:
            word_perfect = 1.0
        return word_accuracy, word_completeness, word_perfect


if __name__ == "__main__":
    trigram = NGrams(VOCAB_PATH, 3)
    trigram.get_corpus()
    trigram.generate_grams()
    trigram.calculate_prob(smoothing_method='laplace')

    bigram = NGrams(VOCAB_PATH, 2)
    bigram.get_corpus()
    bigram.generate_grams()
    bigram.calculate_prob(smoothing_method='laplace')

    bigram.unigram()

    unigram_dict = bigram.unigram_prob
    bigram_prob_df = bigram.prob_df
    trigram_prob_df = trigram.prob_df

    corpus = [['p aʊ ɝ', 'p o w e r']]

    # new_corpus = [[item.split() for item in sublist] for sublist in corpus]
    # for pair in new_corpus:
    #     for phoneme in pair[0]:
    first_letter = phoneme_letter_df.loc['p'].idxmax()
    result_df = phoneme_letter_df.loc[['p', 'aʊ', 'ɝ']].round(3)
    print(result_df)
    row = ['p', 'aʊ', 'ɝ']
    for pho in row:
        positive_columns = result_df.loc[pho][result_df.loc[pho] > 0].index.tolist()
        print(positive_columns)

    # for pho in row:
    #     print(result_df.loc[[pho],[:]>0])
    # 基于bigram的求出p后面跟字母
    # print(bigram_prob_df[first_letter])
    # 基于音标求出所有可能的字母

    # print(sum(unigram_dict.values()))
    # print(bigram_prob_df.sum(axis=1))
    # print(trigram_prob_df.sum(axis=1))

    # 读取音标和字母的数据然后也搞一个df根据这四个表做个预测

    # seq, seq_prob = ngram.calculate_seq_prob('discard')
    # print(f'word: {seq}: prob is {seq_prob}')
    # generated_spelling, answer = ngram.predict('thjkrehgkjehg')
    # print(f'generated spelling is {generated_spelling}; correct answer is {answer}')
    # accuracy, completeness, perf = ngram.evaluation(generated_spelling, answer)
    # print(f'the accuracy is {accuracy}, the completeness is {completeness}, the perfect spelling is {perf}')
