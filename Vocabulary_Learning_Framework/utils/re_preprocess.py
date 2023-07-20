"""
purposes: (1)provide the interface to build costuming data
          (2) save data to json file
要把四种单词信息按顺序保存到文件中，为了避免重复意思，最好保存为字典的形式，这样每一个单词只对应一个意思

1: 不同单词书有其各自的单词，但是【句子，图片，视频，发音】都可以共用，放在各自的总文件夹下，学习任何一个单词都可以直接调用
2：当前项目为了简化，只处理【中文，音标，词性 英文拼写】，其他的信息提供调用方法

Simon说会议的主要重点是：（1）建立interface （2）定义好用的类型 （3）history information是包括过去所有的单词信息

"""
import re
import io
import os
import abc
from phonetic_process import get_phonemes
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
                self._raw_file_path: str, mandatory. for costuming data. raw data path
                self._write_file_path: str, mandatory. write to target file
                self._vocab_data, vocab data, format [chinese, pos, [phonemes], English]  for example
                ['丢弃', 'vt', ['d', 'ɪ', 's', 'k', 'ɑ', 'r', 'd'], 'discard']
                **agent_specific_kwargs: optional extra args.
                """
        self._raw_file_path = vocab_book_raw_data_path
        self._write_file_path = vocab_book_write_path

    @abc.abstractmethod
    def preprocess_book(self):
        """
        write your own code to preprocess data, the order must be in line with [chinese, pos, [phonemes], English]
        # 之所以是这种格式是因为，后期还要选择要什么信息
        for example ['丢弃', 'vt', ['d', 'ɪ', 's', 'k', 'ɑ', 'r', 'd'], 'discard']
        save self._vocab_data to json file
        """


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
        results = {}  # store results，为了去除多意思单词
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
                        _, word_phonetic = get_phonemes(result_list[0])
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

