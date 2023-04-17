"""Tic-tac-toe (noughts and crosses), implemented in Python.
This is a demonstration of implementing a deterministic perfect-information
game in Python.
"""

import numpy as np
from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

_NUM_PLAYERS = 2  # 两个玩家
_NUM_ROWS = 3  # 行数
_NUM_COLS = 3  # 列数
_NUM_CELLS = _NUM_ROWS * _NUM_COLS  # 一共有9个空
_GAME_TYPE = pyspiel.GameType(
    short_name="python_tic_tac_toe",  # 游戏的名字
    long_name="Python Tic-Tac-Toe",  # 游戏的名字
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,  # 有先后顺序的游戏
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,  # 游戏中没有机会玩家
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,  # 玩家知道的信息都是一样的
    utility=pyspiel.GameType.Utility.ZERO_SUM,  # 效用的类型是zero-sum的游戏
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,  # 游戏结束
    max_num_players=_NUM_PLAYERS,  # 最大的玩家数量
    min_num_players=_NUM_PLAYERS,  # 最小的玩家数量
    provides_information_state_string=True,  # 提供string 类型信息
    provides_information_state_tensor=False,  # 提供tensor类型的信息
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})  # 游戏的特殊参数
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CELLS,  # 可以给9个地方下子
    max_chance_outcomes=0,  # 没有机会玩家
    num_players=2,  # 玩家数量
    min_utility=-1.0,  # 输家得分
    max_utility=1.0,  # 赢家得分
    utility_sum=0.0,  # 输赢游戏
    max_game_length=_NUM_CELLS)  # 最大游戏步长是9个位置都满了


class TicTacToeGame(pyspiel.Game):
    """A Python version of the Tic-Tac-Toe game."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return TicTacToeState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        if ((iig_obs_type is None) or
                (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
            return BoardObserver(params)
        else:
            return IIGObserverForPublicInfoGame(iig_obs_type, params)


class TicTacToeState(pyspiel.State):
    """A python version of the Tic-Tac-Toe state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._cur_player = 0  # 当前的游戏玩家
        self._player0_score = 0.0  # 玩家0的得分
        self._is_terminal = False  # 游戏是否结束
        # 生成棋盘
        self.board = np.full((_NUM_ROWS, _NUM_COLS), ".")

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

    # 如果游戏结束则结束，否则返回当前的player
    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    # 可以采取的合法的动作，只有是'.'才能代表是合法的动作数量
    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == "."]

    # 把这个动作画到环境中
    def _apply_action(self, action):
        """Applies the specified action to the state."""
        self.board[_coord(action)] = "x" if self._cur_player == 0 else "o"  # 0玩家是o,1玩家是x 【(2，1)：x】
        if _line_exists(self.board):
            self._is_terminal = True  # 如果游戏已经结束
            self._player0_score = 1.0 if self._cur_player == 0 else -1.0  # 看哪个玩家加分
        elif all(self.board.ravel() != "."):  # 首先将其展开，然后判断棋盘中是否还有'.'，没有的话游戏结束
            self._is_terminal = True
        else:
            self._cur_player = 1 - self._cur_player  # 否则切换玩家

    def _action_to_string(self, player, action):
        """Action -> string."""
        row, col = _coord(action)
        # (player,(row,col)) 主要是得到玩家的每一步操作
        return "{}({},{})".format("x" if player == 0 else "o", row, col)

    # 判断游戏是否结束
    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    # 返回得分
    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return [self._player0_score, -self._player0_score]

    # 打印棋盘
    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return _board_to_string(self.board)


class BoardObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""
    def __init__(self, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        shape = (1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS)
        self.tensor = np.zeros(np.prod(shape), np.float32)  # 生成了9个0
        self.dict = {"observation": np.reshape(self.tensor, shape)}  # 生成了一个observation的表格

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        del player
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]  # 返回的是表格
        obs.fill(0)  # 全部填充为0
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                cell_state = ".ox".index(state.board[row, col])  # cell-state就是返回那个图形
                obs[cell_state, row, col] = 1  # 已经图形 行列  以及标为 1

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        del player
        return _board_to_string(state.board)


# Helper functions for game details.

# 如果某一行的形状一样，则代表游戏结束且赢了，输入是横竖对角线的三个点的坐标，
def _line_value(line):
    """Checks a possible line, returning the winning symbol if any."""
    if all(line == "x") or all(line == "o"):
        return line[0]


# 判断有没有赢 横 竖 对角线
def _line_exists(board):
    """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
    return (_line_value(board[0]) or _line_value(board[1]) or
            _line_value(board[2]) or _line_value(board[:, 0]) or
            _line_value(board[:, 1]) or _line_value(board[:, 2]) or
            _line_value(board.diagonal()) or
            _line_value(np.fliplr(board).diagonal()))


# 返回坐标 输入是采取的动作，输出该动作所在的坐标
def _coord(move):
    """Returns (row, col) from an action id."""
    return (move // _NUM_COLS, move % _NUM_COLS)


def _board_to_string(board):
    """Returns a string representation of the board."""
    return "\n".join("".join(row) for row in board)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, TicTacToeGame)
