"""
Define: the environment interface
Return: TimeStep [observation, reward (uncertain?), discount(uncertain?), step_type]
Observation members :observation = {"vocab_sessions", "current_session_num", "vocab_session", "legal_actions",
                                    "current_player", "condition", "answer", "answer_length", "student_spelling",
                                    "letter_feedback", "accuracy", "completeness", "history"}
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
                 vocab_path: the vocab data path for load vocabulary data
                 vocab_book_name: options [CET4, CET6], the book you want use
                 chinese_setting=True, do you want chinese?
                 phonetic_setting=True, do you want phonetic?
                 POS_setting=True, do you want POS?
                 english_setting=True, must be true

                 new_words_number: the number of words per session
                 self._vocabulary_sessions: randomly split vocabulary data into sessions
                 self._state: read necessary information from state object
                 self._discount: discount !!!!!!!!!!!!!!!!!!!!!!!!!!

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

    @abc.abstractmethod
    def new_initial_state(self):
        """
                :return: initialize the state of the game
                """

    @abc.abstractmethod
    def reset(self):
        """
               :return: construct and return the initial state of game. TimeStep
               """

    @abc.abstractmethod
    def get_time_step(self):
        """
               :return: construct middle state of game. TimeStep
               """

    @abc.abstractmethod
    def step(self, action):
        """
               :return: (1) apply_action,(2) construct middle state of game (3) returns TimeStep
               """
    @property
    def vocabulary_data(self):
        """
                       :return: vocabulary data
                       """
        return self._vocab_data
