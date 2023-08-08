"""
Define: the environment interface
Return TimeStep [observation, reward, discount, step_type]
Observations members :observations = {"vocab_sessions", "current_session_num", "current_player", "legal_actions",
                        "vocab_session",.......}
"""

import abc
from typing import List
from utils.choose_vocab_book import ReadVocabBook
import random


class EnvironmentInterface(metaclass=abc.ABCMeta):
    """ reinforcement learning environment interface base."""

    @abc.abstractmethod
    def __init__(self,
                 vocab_path: str,
                 vocab_book_name: str,
                 chinese_setting: bool,
                 phonetic_setting: bool,
                 POS_setting: bool,
                 english_setting: bool,
                 new_words_number: int,
                 discount: float = 1.0
                 ):
        """
        :args
                 vocab_path: the vocab data path
                 vocab_book_name: options [CET4, CET6], the book you want use
                 chinese_setting=True, do you want chinese?
                 phonetic_setting=True, do you want phonetic?
                 POS_setting=True, do you want POS?
                 english_setting=True, must be true
                 new_words_number: the number of words in one session

                 self._state: store the information state from state object
                 self._discount: discount
                 self._vocabulary_sessions: randomly split vocabulary data into sessions
                 self._should_reset: the timing to reset the game
                 self._player_num: the number of players in my game
                """

        self._vocab_book_name: str = vocab_book_name
        self._ReadVocabBook = ReadVocabBook(vocab_book_path=vocab_path,
                                            vocab_book_name=vocab_book_name,
                                            chinese_setting=chinese_setting,
                                            phonetic_setting=phonetic_setting,
                                            POS_setting=POS_setting,
                                            english_setting=english_setting)
        self._vocab_data = self._ReadVocabBook.read_vocab_book()
        self._vocabulary_sessions: List = []
        random.shuffle(self._vocab_data)
        for i in range(0, len(self._vocab_data), new_words_number):
            vocabulary_session = self._vocab_data[i:i + new_words_number]
            self._vocabulary_sessions.append(vocabulary_session)

        self._state = None
        self._discount = discount
        self._should_reset: bool = True
        self._player_num: int = 4

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
                :return: construct the initial state of the game. TimeStep
                """

    @abc.abstractmethod
    def reset(self):
        """
               :return: Returns the initial state of game, TimeStep
               """

    @abc.abstractmethod
    def get_time_step(self):
        """
               :return: construct middle state of game, TimeStep
               """

    @abc.abstractmethod
    def step(self, action):
        """
               :return: Returns a TimeStep of game, TimeStep
               """

    @property
    def vocab_information_format(self) -> List[str]:
        """
        :return: vocab information format
        """
        return self._vocab_information_format

    @property
    def book_name(self) -> str:
        """
        :return: book name
        """
        return self._vocab_book_name


