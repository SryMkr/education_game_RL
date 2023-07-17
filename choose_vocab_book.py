"""
This file is for choose vocabulary book and information
input[vocabulary name, information]-> output [format information]
"""

import json
import abc
from typing import List


class ReadVocabBookInterface(metaclass=abc.ABCMeta):
    """ a base class to read data"""

    @abc.abstractmethod
    def __init__(self,
                 vocab_book_path: str,
                 vocab_book_name: str,
                 chinese_setting: bool,
                 phonetic_setting: bool,
                 POS_setting: bool,
                 english_setting: bool):
        """
        self._vocab_book_path is the data path
        self._vocab_data is the vocab data
        self._vocab_book_name, name your book or topic
        others are vocab information setting
        """
        self._vocab_book_path = vocab_book_path
        self._vocab_book_name = vocab_book_name
        self._vocab_data: List[List[str, str, List[str], str]] = []
        self._chinese_setting = chinese_setting
        self._phonetic_setting = phonetic_setting
        self._POS_setting = POS_setting
        self._english_setting = english_setting

    @abc.abstractmethod
    def read_vocab_book(self) -> List:
        """
        return the prefer vocab information
        :return: suggested format [chinese, pos, [phonemes], English]
        """
        return self._vocab_data


class ReadVocabBook(ReadVocabBookInterface):
    """ an example to read data"""
    def __init__(self,
                 vocab_book_path,
                 vocab_book_name,
                 chinese_setting,
                 phonetic_setting,
                 POS_setting,
                 english_setting):
        super().__init__(vocab_book_path,
                         vocab_book_name,
                         chinese_setting,
                         phonetic_setting,
                         POS_setting,
                         english_setting)

    def read_vocab_book(self):
        with open(self._vocab_book_path, 'r') as j_file:
            vocabulary_data = json.load(j_file)
            for key, values in vocabulary_data.items():
                word_data = []
                if self._chinese_setting:
                    word_data.append(key)
                if self._POS_setting:
                    word_data.append(values[0])
                if self._phonetic_setting:
                    word_data.append(values[1])
                if self._english_setting:
                    word_data.append(values[2])
                self._vocab_data.append(word_data)
        return self._vocab_data

