"""
define the vocabulary spelling state

state provide whole necessary information to construct the TimeStep of environment

"""

import abc
from typing import List


class StateInterface(metaclass=abc.ABCMeta):
    """The state interface
    :args
        self._vocab_sessions: the vocab sessions from environment
        self._current_session: the current session, change over time
        self._game_over: the game state
        self._current_player_action: the current player
        self._session_data: tasks in one session
        self._legal_actions: construct legal action for each agent

        self._condition: str = '', for student spelling
        self._answer: str = '', for examine
        self._answer_length: int = 0, control the answer length
    """

    @abc.abstractmethod
    def __init__(self, vocab_sessions):
        self._vocab_sessions = vocab_sessions
        self._current_session_num: int = 0
        self._game_over: bool = False
        self._current_player: int = 0
        self._rewards: List[int] = []
        self._vocab_session: List[List[str]] = []
        self._legal_actions: List = [[i for i in range(len(self._vocab_sessions))],
                                     [],
                                     [i for i in range(26)],
                                     [0, 1]]

        self._condition: str = ''
        self._answer: str = ''
        self._answer_length: int = 0

        self._LETTERS: List[str] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                         'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        self._stu_spelling: List[str] = []

        self._letter_feedback: List[int] = []
        self._accuracy: float = 0.0
        self._completeness: float = 0.0

    @property
    def current_player(self) -> int:
        """
        :return: Returns the current player index

        """
        return self._current_player

    @property
    def current_session_num(self) -> int:
        """
        :return: Returns current session.
        """
        return self._current_session_num

    @abc.abstractmethod
    def legal_actions(self, player_ID) -> List:
        """
        # 今天早上的任务是，构造合法的动作空间，并且能够变化
        我的所有的agent都不共享动作空间，所以每一个agent能采取的动作都是固定的，所以在初始化的时候可以试着自己构造
        :return: Returns the legal action of the agent.
        """
        return self._legal_actions[player_ID]

    @abc.abstractmethod
    def apply_action(self, action) -> int:
        """
        apply action
        :return: Returns the (player ID) of the next player.
        """

    @property
    def is_terminal(self) -> bool:
        """
                :return: the game status .
                """
        return self._game_over

    @property
    def rewards(self) -> List[int]:
        """
        :return: Returns the rewards .
        """
        return self._rewards

    @property
    def vocab_sessions(self) -> List:
        """
        :return: Returns the current session tasks.
        """
        return self._vocab_sessions

    @property
    def vocab_session(self) -> List[List[str]]:
        """
        :return: Returns the current session tasks.
        """
        return self._vocab_session

    @property
    def answer_length(self) -> int:
        """
        :return: Returns answer length
        """
        return self._answer_length

    @property
    def answer(self) -> str:
        """
        :return: Returns the correct answer
        """
        return self._answer

    @property
    def condition(self) -> str:
        """
        :return: Returns the condition for spelling
        """
        return self._condition

    @property
    def stu_spelling(self) -> List[str]:
        """
        :return: Returns the condition for spelling
        """
        return self._stu_spelling

    @property
    def letter_feedback(self) -> List[int]:
        """
        :return: Returns the condition for spelling
        """
        return self._letter_feedback

    @property
    def accuracy(self) -> float:
        """
        :return: Returns the condition for spelling
        """
        return self._accuracy

    @property
    def completeness(self) -> float:
        """
        :return: Returns the condition for spelling
        """
        return self._completeness
