"""
purposes: (1)provide the interface to build costuming data
          (2)one example of process data
          (3) save to json file

就是提供一个文件，可以自己处理自己的数据并且保存到文件中，最后保存的的文件格式一样
1: 不同单词书有其各自的单词，但是【句子，图片，视频，发音】都可以共用，放在各自的总文件夹下，学习任何一个单词都可以直接调用
2：当前项目为了简化，只处理【中文，音标，词性 英文拼写】，其他的信息提供调用方法

Simon说会议的主要重点是：（1）建立interface （2）定义好用的类型 （3）history information是包括过去所有的单词信息
所有的输入和输出都有自己的数据类型，以及都有自己的模板

1: 研究人员肯定不需要自己准备数据的，所以只要输入单词书的名字就可以
2：研究人员只需要输入想要的信息就可以得到固定的信息格式 【book name, information】
3：环境里面可以返回单词书以及信息的格式，还有各种有关预处理结果数据的信息  【查看书名，单词信息】



"""
import re
import io
import os
import abc
from phonetic_process import get_phonetic_components
from typing import List
import json


class PreProcessBookInterface(metaclass=abc.ABCMeta):
    """Abstract base class for customizing vocabulary data.
    save vocab data to json file
    """

    @abc.abstractmethod
    def __init__(self,
                 vocab_book_raw_data_path: str,
                 vocab_book_write_path: str,
                 **agent_specific_kwargs):
        """Initialize parameters.

            Args:
                self._raw_file_path: str, mandatory. for costuming data. read from file
                self._write_file_path: str, mandatory. write to file
                self._vocab_data, vocab data, format [chinese, pos, [phonemes], English]  for example
                ['丢弃', 'vt', ['d', 'ɪ', 's', 'k', 'ɑ', 'r', 'd'], 'discard']
                **agent_specific_kwargs: optional extra args.
                """
        self._raw_file_path = vocab_book_raw_data_path
        self._write_file_path = vocab_book_write_path
        self._vocab_data: List[List[str, str, List[str], str]] = []

    @abc.abstractmethod
    def preprocess_book(self):
        """
        write your own code to preprocess data, the order must be in line with [chinese, pos, [phonemes], English]
        for example ['丢弃', 'vt', ['d', 'ɪ', 's', 'k', 'ɑ', 'r', 'd'], 'discard']
        save self._vocab_data to json file
        """
        self._vocab_data = []


class PreProcessBook(PreProcessBookInterface):
    """an example class to get the vocabulary information, preprocess raw vocabulary data"""

    def __init__(self, vocab_book_raw_data_path, vocab_book_write_path):
        super().__init__(vocab_book_raw_data_path, vocab_book_write_path)

    def preprocess_book(self):
        # chinese = pattern = r'[\u4e00-\u9fff]' # chinese character
        symbols = r'[,;&，；.]'  # the separating symbols in my file
        replacement = ' '  # replace separating symbols by space
        search_pattern = r'[()⋯*《》]'  # search special symbols
        part_of_speech = ['vt', 'a', 'n', 'ad', 'vt', 'vi', 'prep', 'pron', 'conj', 'aux']  # all part of speech
        match_pattern = r'[a-z]+\s+[a-z]+\s+[\u4e00-\u9fff]+'  # match my data pattern
        results = {}  # store results
        with io.open(self._raw_file_path, encoding="utf8") as file:
            for line in file:
                line = line.lower().strip()  # lower letter
                line = re.sub(symbols, replacement, line)  # replace separating symbols by space
                if not re.search(search_pattern, line):  # do not need line with special symbols
                    matches = re.match(match_pattern, line)  # match the data pattern
                    if matches:
                        result = matches.group(0)
                        result = re.sub(r'\s+', ' ', result)
                        result_list = result.split(' ')  # en, pos, ch
                        _, word_phonetic = get_phonetic_components(result_list[0])
                        results[result_list[2]] = [result_list[1], word_phonetic, result_list[0]]
        json_str = json.dumps(results)
        with open(self._write_file_path, 'w') as j_file:
            j_file.write(json_str)


if __name__ == '__main__':
    '''for testing'''
    current_path = os.getcwd()
    raw_file_path = os.path.join(current_path, 'vocabulary_books', 'raw_vocab_data.txt')
    write_file_path = os.path.join(current_path, 'vocabulary_books', 'CET6', 'Vocab.json')
    vocab_book = PreProcessBook(raw_file_path, write_file_path)
    vocab_book.preprocess_book()

