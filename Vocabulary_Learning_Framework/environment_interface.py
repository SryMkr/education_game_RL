"""
# 本文件目前只考虑从文件中读取数据的情况，不考虑预处理数据的接口
def the environment interface
class TimeStep(
collections.namedtuple(
        "TimeStep", ["observations", "rewards", "discounts", "step_type"])): # player observation to decide action
        # in my game, tutor agent see the tutor feedback, then decide the difficulty level

"""

import abc
from utils.choose_vocab_book import ReadVocabBook
from typing import List


class EnvironmentInterface(metaclass=abc.ABCMeta):
    """ reinforcement learning environment class."""

    @abc.abstractmethod
    def __init__(self,
                 vocabulary_path: str,
                 vocabulary_book_name: str,
                 chinese_setting: bool,
                 phonetic_setting: bool,
                 POS_setting: bool,
                 english_setting: bool = True):
        """
        :args
                 vocabulary_book_name, options [CET4, CET6], the book you want use
                 chinese_setting=True, do you want chinese?
                 phonetic_setting=True, do you want phonetic?
                 POS_setting=True, do you want POS?
                 english_setting=True, must be true
                """

        self._vocab_book_name: str = vocabulary_book_name
        self._ReadVocabBook = ReadVocabBook(vocab_book_path=vocabulary_path,
                                            vocab_book_name=vocabulary_book_name,
                                            chinese_setting=chinese_setting,
                                            phonetic_setting=phonetic_setting,
                                            POS_setting=POS_setting,
                                            english_setting=english_setting)
        self._vocab_data = self._ReadVocabBook.read_vocab_book()

        def information_format():
            """
            :return: the information format is also the order of your data
            """
            __information_list = []
            if chinese_setting:
                __information_list.append('chinese')

            if POS_setting:
                __information_list.append('pos')

            if phonetic_setting:
                __information_list.append('phonetic')

            if english_setting:
                __information_list.append('english')
            return __information_list

        self._vocab_information_format = information_format()

    @abc.abstractmethod
    def new_initial_state(self):
        """
        :return: Returns the initial start of game.
        """
        pass

    @property
    def vocab_data(self) -> List:
        """

        :return: iterable,  vocabulary data
        """
        return self._vocab_data

    @property
    def vocab_information_format(self) -> List[str]:
        """

        :return: iterable,  vocabulary data
        """
        return self._vocab_information_format

    @property
    def book_name(self) -> str:
        """

        :return: book name
        """
        return self._vocab_book_name

    # def get_time_step(self):
    #     """
    #            老师中的设定是不是应该在环境中，包括老师能看到什么，学生能看到什么？ 包括可用的字母也是环境的？其实感觉老师和环境都可以，因为
    #            环境其实也就是老师一个人的事
    #            :return: Returns a state of game, ["observations", "rewards", "discounts"]
    #            """
    #     pass

    # def step(self, action):
    #     """
    # 环境采取接受了某个动作后， 环境发生变化，以及对这个动作的奖励
    #            :return: Returns a state of game, ["observations", "rewards", "discounts"]
    #            """
    #     pass

    # current ignore this function
    # @abc.abstractmethod
    # def make_py_observer(self):
    #     """Returns an object used for observing game state."""
    #     pass
