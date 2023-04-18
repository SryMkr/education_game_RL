"""Python implementation of iterated prisoner's dilemma.
This is primarily here to demonstrate simultaneous-move games in Python.
"""

import enum
import numpy as np
import pyspiel

_NUM_PLAYERS = 2  # 两个玩家
# 第一个参数，游戏停止的概率，第一个参数游戏长度
_DEFAULT_PARAMS = {"termination_probability": 0.125, "max_game_length": 9999}
_PAYOFF = [[5, 0], [10, 1]]  # 你可以得到多少分

_GAME_TYPE = pyspiel.GameType(
    short_name="python_iterated_prisoners_dilemma",
    long_name="Python Iterated Prisoner's Dilemma",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,  # sequential game,   mean field game
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,  # deterministic, sampled stochastic
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,  # imperfect information
    utility=pyspiel.GameType.Utility.GENERAL_SUM,  # 得分确实是两个人合作或者竞争来的
    reward_model=pyspiel.GameType.RewardModel.REWARDS,  # 返回一个terminal 还是reward
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)  # 这里提供的外部游戏参数到底有什么用？


class Action(enum.IntEnum):
    COOPERATE = 0  # 合作
    DEFECT = 1  # 背叛


class Chance(enum.IntEnum):
    CONTINUE = 0  # chance节点在这是继续玩还是停止么？
    STOP = 1


class IteratedPrisonersDilemmaGame(pyspiel.Game):
    """The game, from which states and observers can be made."""

    # pylint:disable=dangerous-default-value
    def __init__(self, params=_DEFAULT_PARAMS):
        max_game_length = params["max_game_length"]  # 总共循环多少轮游戏
        super().__init__(
            _GAME_TYPE,  # 第一个参数
            pyspiel.GameInfo(
                num_distinct_actions=2,  # 【合作，背叛】
                max_chance_outcomes=2,  # 【continue，defect】
                num_players=2,
                min_utility=np.min(_PAYOFF) * max_game_length,  # 效用的最小值
                max_utility=np.max(_PAYOFF) * max_game_length,  # 效用的最大值
                utility_sum=None,  # 这是一个general-sum的游戏
                max_game_length=max_game_length,  # 最大游戏步长
            ),
            params,
        )
        self._termination_probability = params["termination_probability"]  # 游戏结束的概率

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return IteratedPrisonersDilemmaState(self, self._termination_probability)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return IteratedPrisonersDilemmaObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)


# 在这里设定state的转换
class IteratedPrisonersDilemmaState(pyspiel.State):
    """Current state of the game."""

    def __init__(self, game, termination_probability):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._current_iteration = 1  # 代表第一轮游戏
        self._termination_probability = termination_probability
        self._is_chance = False  # 不是chance player的回合
        self._game_over = False  # 游戏是否结束
        self._rewards = np.zeros(_NUM_PLAYERS)  # rewards的列表
        self._returns = np.zeros(_NUM_PLAYERS)  # 总rewards的列表

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every simultaneous-move game with chance.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            # 游戏结束
            return pyspiel.PlayerId.TERMINAL
        elif self._is_chance:
            # 该chance player 他来决定是否继续游戏
            return pyspiel.PlayerId.CHANCE
        else:
            # 在SIMULTANEOUS游戏类中，就是使用这个参数控制游戏进程
            return pyspiel.PlayerId.SIMULTANEOUS

    # 定义合法的动作
    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        assert player >= 0  # 至少得有玩家 以下是两个可以采取的动作
        return [Action.COOPERATE, Action.DEFECT]

    # 机会玩家 控制继续游戏，还是停止游戏
    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        assert self._is_chance  # 如果是机会玩家，有一定的概率决定是停止还是继续游戏
        return [(Chance.CONTINUE, 1 - self._termination_probability),
                (Chance.STOP, self._termination_probability)]

    # 这是机会玩家的动作
    def _apply_action(self, action):
        """Applies the specified action to the state."""
        # This is not called at simultaneous-move states.
        assert self._is_chance and not self._game_over
        self._current_iteration += 1  # 如果是机会玩家，迭代次数肯定要加1
        self._is_chance = False   # 紧接着就不是机会玩家了
        self._game_over = (action == Chance.STOP)  # 如果机会玩家选择结束游戏，那么将结束游戏返回
        if self._current_iteration > self.get_game().max_game_length():  # 超过最大的循环次数也要结束游戏
            self._game_over = True

    # 这是游戏玩家的动作
    def _apply_actions(self, actions):
        """Applies the specified actions (per player) to the state."""
        assert not self._is_chance and not self._game_over
        self._is_chance = True  # 接下来该机会玩家出马了
        # 左右两个参数分别是两个玩家各自采取的动作
        self._rewards[0] = _PAYOFF[actions[0]][actions[1]]
        self._rewards[1] = _PAYOFF[actions[1]][actions[0]]
        self._returns += self._rewards  # 将玩家各自的奖励叠加

    # 用string 表示各自的决定
    def _action_to_string(self, player, action):
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return Chance(action).name
        else:
            return Action(action).name

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def rewards(self):
        """Reward at the previous step."""
        return self._rewards

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return self._returns

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return (f"p0:{self.action_history_string(0)} "
                f"p1:{self.action_history_string(1)}")

    def action_history_string(self, player):
        return "".join(
            self._action_to_string(pa.player, pa.action)[0]
            for pa in self.full_history()
            if pa.player == player)


class IteratedPrisonersDilemmaObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        assert not bool(params)
        self.iig_obs_type = iig_obs_type
        self.tensor = None
        self.dict = {}

    def set_from(self, state, player):
        pass

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        if self.iig_obs_type.public_info:
            return (f"us:{state.action_history_string(player)} "
                    f"op:{state.action_history_string(1 - player)}")
        else:
            return None


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, IteratedPrisonersDilemmaGame)
