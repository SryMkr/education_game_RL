"""
def the environment
1: get the game setting
2:
"""

from typing import List, Tuple, Dict
import abc


class Environment(metaclass=abc.ABCMeta):
    """ reinforcement learning environment class."""

    def __init__(self, tasks: Dict[str, str], total_game_round: int):
        self._tasks_pool: Dict[str, str] = tasks
        self._total_game_round: int = total_game_round

    @abc.abstractmethod
    def new_initial_state(self):
        """
        :return: Returns a state corresponding to the start of a game.
        """
        pass

    # current ignore this function
    # @abc.abstractmethod
    # def make_py_observer(self):
    #     """Returns an object used for observing game state."""
    #     pass
