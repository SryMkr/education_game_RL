"""
target: 可以根据音标将字母区分开来，达到完全百分之百的完全拼写概率，根据一个音标对应的字母的概率建立模型，使用概率实现区分
1: 一般情况下，单词的长度不会小于音标的个数
2: 根据单词的拼写，找出每一个字母对应最大可能的音标
3: 看看能不能将一个按照音标的方式直接切割
"""

import pandas as pd
import os
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook

_CURRENT_PATH = os.getcwd()  # get the current path

# ----------------------------------------读取音标字母文件--------------------------------------------------------------
_phoneme_letter_abs_path = os.path.join(_CURRENT_PATH,
                                        'test_data/phoneme_letter_pair/IBM_phoneme_letter_pair.xls')  # get the vocab data path
phoneme_letter_df = pd.read_excel(_phoneme_letter_abs_path, index_col=0, header=0)
# print(phoneme_letter_df)

# ----------------------------------------读取单词文件--------------------------------------------------------------

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
#
#
print(corpus)
# ----------------------------------------实现拼写--------------------------------------------------------------
corpus = [['d', 'ɪ', 's', 'k', 'ɑ', 'r', 'd'], ['d', 'i', 's', 'c', 'a', 'r', 'd']], [['p', 'aʊ', 'ɝ'], ['p', 'o', 'w', 'e', 'r']]

def connect_adjacent_index(nums):
    result = []
    current_group = [nums[0]]

    for i in range(1, len(nums)):
        if nums[i] - nums[i - 1] == 1:
            current_group.append(nums[i])
        else:
            result.append(current_group)
            current_group = [nums[i]]

    result.append(current_group)
    return result


phoneme_letter = {}

for phonemes, letters in corpus:
    # 如果长度相等，代表了一个音标对应一个字母，所以不用管，无非就是拼写对错的问题
    if len(phonemes) == len(letters) or len(phonemes) == 0:
        continue
    else:
        result_df = phoneme_letter_df.loc[phonemes, letters]
        print('corpus is:', phonemes, letters)
        print('Phoneme-Letter prob matrix:\n', result_df)
        max_prob_pairs = result_df.idxmax()
        # print(max_prob_pairs)
        letters = max_prob_pairs.index.tolist()  # 获得字母
        phonemes = max_prob_pairs.values.tolist()  # 获得音标
        # 按照音素合并字母,不能直接合并，应该是按照顺序合并，如果不一样单独列出来，
        # print(letters, phonemes)
        # 音标，索引
        phoneme_index_dict = {}
        for i, phoneme in enumerate(phonemes):
            if phoneme not in phoneme_index_dict:
                phoneme_index_dict[phoneme] = [i]
            else:
                phoneme_index_dict[phoneme].append(i)

        # 相邻的索引合并，并将相邻的索引对应的字母找出来合并在一起
        phoneme_connected_index = {}
        for index, values in phoneme_index_dict.items():
            connected_adj_index = connect_adjacent_index(values)
            phoneme_connected_index[index] = connected_adj_index
        # print(phoneme_connected_index)

        # 将每一个列表的找到对应的字母列表中的字母，该合并的合并，这样直接得到音标对应的一对一和一对多的关系
        pho_let_result = {}
        for phoneme, indexes in phoneme_connected_index.items():
            pho_let_result[phoneme] = [[letters[idx] for idx in sublist] for sublist in indexes]
        print('most likely combo:', pho_let_result)
        # 音标和字母的对应关系要全部弄出来{音标：【字母或者字母组合】}
        for phoneme, indexes in pho_let_result.items():
            letter_pairs = []
            for index in indexes:
                letter_pairs.append(''.join(index))
            if phoneme not in phoneme_letter.keys():
                phoneme_letter[phoneme] = letter_pairs
            else:
                phoneme_letter[phoneme].extend(letter_pairs)

letters = []
for key, value in phoneme_letter.items():
    phoneme_letter[key] = set(value)
    for i in set(value):
        letters.append(i)


