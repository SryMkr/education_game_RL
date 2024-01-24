"""
define the vocabulary spelling state

state provide whole necessary information to help env construct the TimeStep

"""

import abc
from typing import List, Tuple


class StateInterface(metaclass=abc.ABCMeta):
    """The state interface
    :args
        self._vocab_sessions: the vocab sessions from environment
        self._current_session_num: integer, the current session number
        self._current_session: the current session, change over time
        self._game_over: the game state
        self._current_player: the current player
        self._vocab_session: tasks in one session
        self._legal_actions: construct legal action for each agent

        self._condition: str = '', for student spelling
        self._answer: str = '', for examine
        self._answer_length: int = 0, control the answer length
        self._LETTERS: = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                         'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        self._stu_spelling: List[str], store student spelling,

        self._letter_feedback: List[int], student spelling feedback of per letter
        self._accuracy: student answer accuracy
        self._completeness: student answer feedback
    """

    @abc.abstractmethod
    def __init__(self, vocab_sessions):
        self._vocab_sessions = vocab_sessions
        self._current_session_num: int = 1
        self._game_over: bool = False
        self._current_player: int = 0
        self._rewards: int = 0
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

        self._examiner_feedback: Tuple[List[int], float, float] = tuple()
        self._history = []  # what information does the history have?,搞个准确度和完整度吧

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
        :return: Returns the legal action of the agent.
        """
        return self._legal_actions[player_ID]

    @abc.abstractmethod
    def apply_action(self, action) -> int:
        """
        apply action, and store necessary information
        :return: Returns the (player ID) of the next player.
        """

    @abc.abstractmethod
    def reward_function(self, information) -> int:
        """
        apply action, and store necessary information
        :return: Returns the (player ID) of the next player.
        """
        return self._rewards

    @property
    def is_terminal(self) -> bool:
        """
                :return: the game status .
                """
        return self._game_over

    @property
    def rewards(self) -> int:
        """
        :return:  the rewards .
        """
        return self._rewards

    @property
    def vocab_sessions(self) -> List:
        """
        :return: Returns the all session tasks.
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
        :return: student spelling
        """
        return self._stu_spelling

    @property
    def examiner_feedback(self) -> Tuple[List[int], float, float]:
        """
        :return: feedback per letter
        """
        return self._examiner_feedback

    @property
    def history(self) -> List:
        """
        :return: return the history!!!!!!!!!!!!!!!!!!!!!!!!
        """
        return self._history
