"""
def the environment interface
class TimeStep(
collections.namedtuple(
        "TimeStep", ["observations", "rewards", "discounts", "step_type"])): # player observation to decide action
        # in my game, tutor agent see the tutor feedback, then decide the difficulty
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
        :return: Returns the initial start of game.
        """
        pass

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
