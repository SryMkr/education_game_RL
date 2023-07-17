"""
define word maker game environment
"""
from environment_interface import EnvironmentInterface
import os


class VocabSpellGame(EnvironmentInterface):
    def __init__(self,
                 vocabulary_book_path,
                 vocabulary_book_name,
                 chinese_setting,
                 phonetic_setting,
                 POS_setting,
                 english_setting: bool = True):
        super().__init__(vocabulary_book_path,
                         vocabulary_book_name,
                         chinese_setting,
                         phonetic_setting,
                         POS_setting,
                         english_setting)

    def new_initial_state(self):
        pass


if __name__ == '__main__':
    '''testing interface'''
    current_path = os.getcwd()
    vocabulary_absolute_path = os.path.join(current_path, 'vocabulary_books', 'CET4',
                                            'Vocab.json')
    a = VocabSpellGame(vocabulary_absolute_path, 'CET4', True, False, False)
    print(a.book_name, a.vocab_information_format)
    print(a.vocab_data)
