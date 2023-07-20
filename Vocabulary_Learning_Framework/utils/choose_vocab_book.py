"""
This file is for choose vocabulary book and what information you want
input[vocabulary name, information]-> output [information, target]
1: 研究人员肯定不需要自己准备数据的，所以只要输入单词书的名字就可以
2：研究人员只需要输入想要的信息就可以得到固定的信息格式 【book name, information】
3：环境里面可以返回单词书以及信息的格式，还有各种有关预处理结果数据的信息  【查看书名，单词信息】
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
        self._vocab_data is the [information, target]
        self._vocab_book_name: name your book or topic
        others are vocab information setting
        """
        self._vocab_book_path = vocab_book_path
        self._vocab_book_name = vocab_book_name
        self._vocab_data: List[List[str]] = []
        self._chinese_setting = chinese_setting
        self._phonetic_setting = phonetic_setting
        self._POS_setting = POS_setting
        self._english_setting = english_setting

    @abc.abstractmethod
    def read_vocab_book(self) -> List[List[str]]:
        """
        :return: suggested format ['chinese pos phonemes', 'English'] -------[information, target]
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
                    word_data.append(' '.join(values[1]))
                if self._english_setting:
                    word_data.append(values[2])
                information_target = [' '.join(word_data[:-1]), ' '.join(word_data[-1])]
                self._vocab_data.append(information_target)
        return self._vocab_data

