"""
def the environment interface

"""

from typing import Dict
import abc


class Environment(metaclass=abc.ABCMeta):
    """ reinforcement learning environment class."""

    def __init__(self, tasks: Dict[str, str], total_game_round: int):
        """
        :args
                self._tasks_pool: ,mandatory, tasks list
                self._total_game_round, int, mandatory, total game rounds
                """

        self._tasks_pool: Dict[str, str] = tasks
        self._total_game_round: int = total_game_round

    @abc.abstractmethod
    def new_initial_state(self):
        """
        :return: Returns the start of game.
        """
        pass

    # current ignore this function
    # @abc.abstractmethod
    # def make_py_observer(self):
    #     """Returns an object used for observing game state."""
    #     pass
