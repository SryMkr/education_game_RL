"""
Define: the environment interface
Return: TimeStep [observation, reward (uncertain?), discount(uncertain?), step_type]
Observation members :observation = {"vocab_sessions", "vocab_sessions_num", "current_session_num", "vocab_session", "legal_actions",
                                    "current_player", "condition", "answer", "answer_length", "student_spelling",
                                    "examiner_feedback", "history"}

The functions of the environment comprise three parts
(1) get the action from agents, change the old state to a new state
(2) summarize the information from new state, and give it to the agent as the observation
(3) other complementary functions

"""

import abc
from utils.choose_vocab_book import ReadVocabBook
import random


class EnvironmentInterface(metaclass=abc.ABCMeta):
    """ reinforcement learning environment interface."""

    @abc.abstractmethod
    def __init__(self,
                 vocab_path: str,
                 vocab_book_name: str,
                 chinese_setting: bool,
                 phonetic_setting: bool,
                 POS_setting: bool,
                 english_setting: bool,
                 history_words_number: int,
                 review_words_number: int,
                 sessions_number: int,
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

                 history_words_number: how many words in the student agent memory
                 review_words_number: the number of words per session
                 sessions_number:  the session numbers you want (days)

                 self._history_words: the specific history words
                 self._state: read necessary information from state object
                 self._discount: the discount for the reinforcement learning

                 self._should_reset: the timing to reset the game
                 self._player_num: the number of players in my game
                """

        self._vocab_book_name = vocab_book_name  # means the vocabulary book name
        self._history_words_number = history_words_number  # the words number in the past days
        self._review_words_number = review_words_number  # the reviewing words number per session
        self._sessions_number = sessions_number  # the session numbers you want

        self._ReadVocabBook = ReadVocabBook(vocab_book_path=vocab_path,
                                            vocab_book_name=vocab_book_name,
                                            chinese_setting=chinese_setting,
                                            phonetic_setting=phonetic_setting,
                                            POS_setting=POS_setting,
                                            english_setting=english_setting)
        self._vocab_data = self._ReadVocabBook.read_vocab_book()  # get all the vocabulary book words around 3000

        # select the history words in accord with the history words number
        self._history_words = random.sample(self._vocab_data, self._history_words_number)

        self._history_information = {}
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
    def history_words(self):
        """
                       :return: vocabulary data
                       """
        return self._history_words

    @property
    def review_words_number(self):
        """

        :return: the review words number in each session
        """
        return self ._review_words_number

    @property
    def sessions_number(self):
        """

        :return: the number of session, presents the time units (days)
        """
        return self._sessions_number


