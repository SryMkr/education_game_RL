"""
define the word maker state
# 其实就是每一个state应该可以知道什么信息
"""

import abc
from typing import List, Tuple, Dict
import enum
import collections


# define four player ID
class PlayerID(enum.IntEnum):
    CHANCE = 0
    TUTOR = 1
    STUDENT = 2
    EXAMINER = 3
    TERMINAL = 4


_PLAYER_ACTION = collections.namedtuple('PlayerAction', ['player', 'action'])


class StateInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, tasks: Dict[str, str], total_game_round: int = 4):
        self._tasks_pool: Dict[str, str] = tasks
        self._total_game_round: int = total_game_round
        self._game_over: bool = False
        # self._next_player: int = 0
        self._current_game_round: int = 1  # the initial game round
        self._player_action: _PLAYER_ACTION = _PLAYER_ACTION('chance', 'select_word')

    @property
    def current_player(self) -> str:
        """
        :return: Returns the player name of the acting player.
        """
        return self._player_action.player

    @property
    def current_game_round(self) -> int:
        """
        :return: Returns the player ID of the acting player.
        """
        return self._current_game_round

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
    def is_terminal(self) -> bool:
        return self._game_over

    # @property
    # def rewards(self) -> int:
    #     """
    #             :return: Returns the rewards of the tutor agent.
    #             """
    #     pass
