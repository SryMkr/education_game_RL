"""
This code is IBM model 1 for statistical translation model
build the pair of phoneme-letter
"""

import math
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook
import os
import pandas as pd

_CURRENT_PATH: str = os.getcwd()  # get the current path

_vocab_book_abs_path: str = os.path.join(_CURRENT_PATH, 'vocabulary_books', 'CET4',
                                         'Vocab.json')  # get the vocab data path

# initialize vocabulary instance
_vocab_instance = ReadVocabBook(vocab_book_path=_vocab_book_abs_path,
                                vocab_book_name='CET4',
                                chinese_setting=False,
                                phonetic_setting=True,
                                POS_setting=False,
                                english_setting=True)

# read vocab data [[phoneme, english]......] [['j ɔ r', 'y o u r'], ['j ɝ s ɛ l f', 'y o u r s e l f']......]
_VOCAB_DATA = _vocab_instance.read_vocab_book()
# print(_VOCAB_DATA)

corpus = [[item.split() for item in sublist] for sublist in _VOCAB_DATA]
# print(corpus)
# corpus = [[['一本', '书'], ['a', 'book']], [['一本', '杂志'], ['a', 'magazine']]]
# corpus = [[['w', 'i', 'k'], ['w', 'e', 'a', 'k']], [['w', 'i', 'k', 'ʌ', 'n'], ['w', 'e', 'a', 'k', 'e', 'n']]]

# set the english vocabulary and foreign language vocabulary
english_vocab = []
foreign_vocab = []

for sp in corpus:
    for fw in sp[0]:  # the foreign language corpus
        foreign_vocab.append(fw)
    for ew in sp[1]:  # the English language corpus
        english_vocab.append(ew)

# covert into lower letter, and omit the duplicated word
english_words = sorted(list(set(english_vocab)), key=lambda s: s.lower())
foreign_words = sorted(list(set(foreign_vocab)), key=lambda s: s.lower())


# print('English words:\n', len(english_words))
# print('Foreign words:\n', len(foreign_words))

'''
# 给定e,f句子和t,计算p(e|f)
def probability_e_f(e, f, t, epsilon=1):
    l_e = len(e)
    l_f = len(f)
    p_e_f = 1
    for ew in e:
        t_ej_f = 0
        for fw in f:
            t_ej_f += t[fw][ew]
        p_e_f = t_ej_f * p_e_f
    p_e_f = p_e_f * epsilon / ((l_f + 1) ** l_e)
    return p_e_f


# 输入语料库计算perplexity
def perplexity(corpus, t, epsilon=1):
    min_prob = 1
    log2pp = 0
    for sp in corpus:
        prob = probability_e_f(sp[1], sp[0], t)
        if prob <= min_prob and prob != 0:
            min_prob = prob
        if prob == 0:
            prob = min_prob

        log2pp += math.log(prob, 2)

    pp = 2.0 ** (-log2pp)
    return pp
'''

t = {}  # 保存所有的翻译概率
init_val = 1.0 / len(english_words)  # 初始情况下，一个外文单词翻译成不同英文单词的概率是相同的

for fw in foreign_words:
    for ew in english_words:
        if fw not in t:
            t[fw] = {}
        t[fw][ew] = init_val

# print('\nInit t', t)


num_epochs = 20
s_total = {}

for epoch in range(num_epochs):
    print("--------epoch % s--------" % (epoch + 1))
    # perplexities.append(perplexity(corpus, t))
    count = {}
    total = {}

    # initialize the counts of fw-ew pair
    for fw in foreign_words:
        total[fw] = 0.0
        for ew in english_words:
            if fw not in count:
                count[fw] = {}
            count[fw][ew] = 0.0
    # print('fw-ew pairs: ', count)

    for sp in corpus:
        # 对于每一个corpus，将所有的外文单词对应到每一个英文单词上面的概率值相加，相当于不同的对齐方式
        for ew in sp[1]:
            s_total[ew] = 0.0
            for fw in sp[0]:
                s_total[ew] += t[fw][ew]

        for ew in sp[1]:
            # 将每一种对齐关系中的概率除于总的概率（所有对齐关系）近似于求出每一种对齐方式的概率，相加就是对应关系的期望数量
            for fw in sp[0]:
                count[fw][ew] += t[fw][ew] / s_total[ew]  # 对于任何一种对齐方式  求出其期望数量
                total[fw] += t[fw][ew] / s_total[ew]  # 求出不同外文单词的概率

    # 对整个表格进行归一化处理
    for fw in foreign_words:
        for ew in english_words:
            t[fw][ew] = count[fw][ew] / total[fw]

    phoneme_letter_df = pd.DataFrame()
    for fw in t:
        # print('foreign word: ', fw)
        sorted_list = sorted(t[fw].items(), key=lambda x: x[1], reverse=True)
        for (ew, p) in sorted_list:
            # print('prob to %s \tis %f' % (ew, p))
            phoneme_letter_df.loc[fw, ew] = round(p, 5)
            phoneme_letter_df = phoneme_letter_df.fillna(0)
    phoneme_letter_df.to_excel('test_data/phoneme_letter_pair/IBM_phoneme_letter_pair.xls')



