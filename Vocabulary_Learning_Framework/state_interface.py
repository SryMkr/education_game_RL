"""
define the word maker state
每一个agent的observation是环境给的，是我的环境包括所有的信息，agent自己去挑，还是环境根据agent的不同，返回state
open spiel中对于private信息是怎么定义的，怎么给不同的玩家不同的observation的？
"""

import abc
from typing import List, Tuple, Dict
from agents_instance import SessionCollectorPlayer

import collections

_PLAYER_ACTION = collections.namedtuple('PlayerAction', ['player', 'action'])


# (1) 知道所有的单词，（2）知道当前的session （3）知道当前的任务
class StateInterface(metaclass=abc.ABCMeta):
    """The state interface
    :args
        self._vocab_data: the vocabulary book data
        self._current_session: the current session, change over time
        self._game_over: the game state
    """

    @abc.abstractmethod
    def __init__(self, vocab_data):
        self._vocab_data = vocab_data
        self._current_session: int = 0
        self._game_over: bool = False
        self._session_player = SessionCollectorPlayer(player_id=0, player_name='session_player',
                                                      vocabulary_data=self._vocab_data,
                                                      new_words_number=10, new_selection_method='sequential')
        self._session_player.piece_data()  # 至此才把单词的分组弄完了
        self._session_data: List[List[str]] = []
        self._player_action: _PLAYER_ACTION = _PLAYER_ACTION('session_player', 'session_collect')

    @property
    def current_player(self) -> str:
        """
        :return: Returns the player name of the acting player.
        """
        return self._player_action.player

    @property
    def current_session(self) -> int:
        """
        :return: Returns current session.
        """
        return self._current_session

    @property
    def legal_action(self) -> str:
        """
        :return: Returns the action of the current agent.
        """
        return self._player_action.action

    @abc.abstractmethod
    def apply_action(self, action: str) -> _PLAYER_ACTION:
        """
        apply action
        :return: Returns the (player ID, action name) of the next player.
        """
        pass

    @property
    def vocab_data(self) -> Dict[str, str]:
        return self._vocab_data

    #
    # @property
    # def total_game_round(self) -> int:
    #     return self._total_game_round

    @property
    def is_terminal(self) -> bool:
        return self._game_over

    # @property
    # def rewards(self) -> int:
    #     """
    #             :return: Returns the rewards of the tutor agent.
    #             """
    #     pass
