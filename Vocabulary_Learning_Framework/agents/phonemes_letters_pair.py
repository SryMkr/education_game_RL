import os
from typing import List, Dict
from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook
from collections import Counter
import pandas as pd

# ----------------------------------------------read vocabulary data---------------------------------------------------------------
CURRENT_PATH = os.getcwd()  # get the current path
# concatenate the vocab data path
_vocabulary_absolute_path = os.path.join(CURRENT_PATH, 'vocabulary_books', 'CET4', 'Vocab.json')
# initialize vocabulary instance
_vocab_instance = ReadVocabBook(vocab_book_path=_vocabulary_absolute_path,
                                vocab_book_name='CET4',
                                chinese_setting=False,
                                phonetic_setting=True,
                                POS_setting=False,
                                english_setting=True)

# read vocab data [[phonemes, english]......] [['j ɔ r', 'y o u r'], ['j ɝ s ɛ l f', 'y o u r s e l f']]
_vocab_data: List[List[str]] = _vocab_instance.read_vocab_book()
# print(_vocab_data)

#  Count the occurrences of each pair of single phoneme and letter in the given string.
def count_pairs(phonemes: str, letters: str, counter: Counter) -> [str, int]:
    """
    Count the occurrences of each pair of phonemes and letters in the given string.
    """
    phonemes: List[str] = phonemes.split()
    for phoneme in phonemes:
        if phoneme == ' ':
            break
        else:
            letters = ''.join(letters.split())  # remove the space between letters
            for letter in letters:
                pair = (phoneme, letter)
                counter[pair] += 1

    return counter


if __name__ == '__main__':
    pair_counter = Counter()
    for phonetic, word in _vocab_data:
        count_pairs(phonetic, word, pair_counter)
    # print(pair_counter)
    # use pandas create matrix
    phoneme_letter_df = pd.DataFrame()
    for phoneme_letter_pair, frequency in pair_counter.items():
        phoneme_letter_df.loc[phoneme_letter_pair[0], phoneme_letter_pair[1]] = frequency
    phoneme_letter_df = phoneme_letter_df.fillna(0)
    phoneme_letter_df.to_excel('test_data/phoneme_letter_pair/phoneme_letter_pair.xls')
    print(phoneme_letter_df)

# 如何使用这个做出马尔可夫过程得拼写过程，展示所有的拼写的可能

