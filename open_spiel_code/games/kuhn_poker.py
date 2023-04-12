import enum
import numpy as np
import pyspiel


# 定义玩家可以有两个动作，一个是pass一个是bet
class Action(enum.IntEnum):
    PASS = 0
    BET = 1


_NUM_PLAYERS = 2  # 定义了玩家的个数
_DECK = frozenset([0, 1, 2])  # 定义了三张牌 返回一个冻结的集合，不能对集合做任何修改，只能调用
_GAME_TYPE = pyspiel.GameType(
    short_name="python_kuhn_poker",  # 环境名字
    long_name="Python Kuhn Poker",
    # 动态一共有三种类型：1：sequential 玩家有先后顺序 kuhn_poker 2: simultaneous game (石头剪刀布) 3：mean field(1 vs N)
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,  # 玩家有先后顺序
    # deterministic：没有机会玩家； EXPLICIT_STOCHASTIC 游戏中有机会玩家,概率已知；Sampled_STOCHASTIC,游戏中有机会玩家,概率未知
    # 这个东西在扑克牌的例子中就是美丽荷官在线发牌，在这个例子中概率是已知的
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    # one-shot; PERFECT_INFORMATION，玩家观察到的信息一样，IMPERFECT_INFORMATION，玩家观察到的信息不一样
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,  # 不完全信息的游戏
    # identical表示玩家的回报是一样的，对称的（win +1, loss,-1, tie,0）
    # general_sum 所有玩家的收益总和并不确定，不限制与一样，或者常数，或者为0
    utility=pyspiel.GameType.Utility.ZERO_SUM,  # zero_sum,, constant_sum,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,  # 返回rewards，或者返回terminal
    # 还有playerID的参数可以设定，可以直接调用
    max_num_players=_NUM_PLAYERS,  # 最大的玩家个数
    min_num_players=_NUM_PLAYERS,  # 最小的玩家个数
    provides_information_state_string=True,  # 是否提供information
    provides_information_state_tensor=True,  # 是否提供information
    provides_observation_string=True,  # 是否提供information
    provides_observation_tensor=True,  # 是否提供information
    provides_factored_observation_string=True)  # 是否提供information

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(Action),  # 动作的个数
    max_chance_outcomes=len(_DECK),  # 在该例子当中，就是机会玩家手里有多少牌
    num_players=_NUM_PLAYERS,  # 最多的玩家个数
    min_utility=-2.0,  # 最小得分
    max_utility=2.0,  # 最大得分
    utility_sum=0.0,  # 效用总和为0，因为是个零和游戏
    max_game_length=3)  # e.g. Pass, Bet, Bet # 最大的游戏步长


class KuhnPokerGame(pyspiel.Game):
    """A Python version of Kuhn poker."""

    def __init__(self, params=None):
        # 如果params不是None或者空字典，作为字典初始化，否则，使用空字典代替，以保证params参数始终被看作为字典
        # 前面的两个形式参数已经在之前定义好了，有一堆参数可以调用
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):  # 返回游戏最开始的状态，并且可以通过返回的类调用state所有想知道的值
        """Returns a state corresponding to the start of a game."""
        return KuhnPokerState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return KuhnPokerObserver(iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params)


# 定义这个游戏的state可能存在的所有信息
class KuhnPokerState(pyspiel.State):
    """A python version of the Kuhn poker state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self.cards = []  # 记录玩家手里的牌，先给A发牌再给B发牌
        self.bets = []  # 用来记录trajectory，最长就3个
        self.pot = [1.0, 1.0]  # 每个玩家的赌注总额度
        self._game_over = False  # 判断游戏是否结束，初始化的时候游戏肯定没结束
        self._next_player = 0

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:  # 如果游戏结束
            # 返回游戏结束的ID
            return pyspiel.PlayerId.TERMINAL
            # 发牌阶段还没结束，返回机会玩家的ID
        elif len(self.cards) < _NUM_PLAYERS:
            return pyspiel.PlayerId.CHANCE  # 这个chance是什么，可能是发牌
        else:  # 如果发牌结束，而且游戏没结束，开始完游戏
            return self._next_player

    def _legal_actions(self, player):  # 输入当前是哪个玩家，以及能够采取的动作
        """Returns a list of legal actions, sorted in ascending order."""
        assert player >= 0  # 用于判断一个表达式，结果为false的时候直接触发异常
        return [Action.PASS, Action.BET]  # 返回两个可采取的动作，返回的可不是值

    def chance_outcomes(self):  # 机会玩家可以采取的动作，及其概率
        """Returns the possible chance outcomes and their probabilities."""
        assert self.is_chance_node()  # 是不是发牌的阶段
        outcomes = sorted(_DECK - set(self.cards))  # 牌库的牌-发出去的牌
        p = 1.0 / len(outcomes)  # 如果牌库里三张牌，那么每张牌的概率是1/3，如果只剩两张概率变为1/2
        return [(o, p) for o in outcomes]  # 返回所有可能的（card, probability）

    def _apply_action(self, action):  # 游戏中一共有三个玩家，要确定是哪个玩家，以及可以采取的动作及动作的概率
        """Applies the specified action to the state."""
        if self.is_chance_node():  # 如果是美丽荷官发牌员
            self.cards.append(action)  # 就是随机发一张牌
        else:
            self.bets.append(action)  # 如果玩家已经开始玩了，那就要记录玩家采取了什么动作，一共5种可能
            if action == Action.BET:  # 如果bet
                self.pot[self._next_player] += 1  # 只要bet 就要给资金池对应的玩家的赌注+1
            self._next_player = 1 - self._next_player  # 切换玩家
            if ((min(self.pot) == 2) or  # 两个人都加注了【bet，bet】
                    (len(self.bets) == 2 and action == Action.PASS) or  # 【bet pass】[pass,pass]
                    (len(self.bets) == 3)):  # [pass,bet,pass] [pass,bet,bet]
                self._game_over = True  # 游戏结束

    def _action_to_string(self, player, action):  # 输入玩家和采取的动作
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:  # 如果还是发牌员，现在还是平手
            return f"Deal:{action}"
        elif action == Action.PASS:  # 如果动作是pass就返回pass
            return "Pass"
        else:  # 如果动作是bet就返回bet
            return "Bet"

    def is_terminal(self):  # 游戏是否结束
        """Returns True if the game is over."""
        return self._game_over

    def returns(self):  # 返回一轮结束后的输赢奖励
        """Total reward for each player over the course of the game so far."""
        pot = self.pot  # 每个玩家的赌注【a,b】
        winnings = float(min(pot))  # 获得赌注的最小值
        if not self._game_over:
            # 游戏还没结束都是【0，0】
            return [0., 0.]
        elif pot[0] > pot[1]:  # action:【1，0】pot:【2，1】说明a赢了,那么a加1分，b减1分
            return [winnings, -winnings]
        elif pot[0] < pot[1]:  # action:【0，1】 pot:【1，2】 说明b赢了,那么b加1分，a减1分
            return [-winnings, winnings]
        elif self.cards[0] > self.cards[1]:  # 另外的三种情况都是要比较卡牌大小的，资金池的钱都一样
            return [winnings, -winnings]
        else:
            return [-winnings, winnings]

    def __str__(self):  # 前两位返回的是AB两个玩家手牌，后面的返回的是两个玩家决策路径
        """String for debug purposes. No particular semantics are required."""
        return "".join([str(c) for c in self.cards] + ["pb"[b] for b in self.bets])


class KuhnPokerObserver:  # 游戏的观察者，可能是返回某一个observation
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:  # 游戏开始observation没有任何参数，如果有就要抛出一个异常
            raise ValueError(f"Observation parameters not supported; passed {params}")

        # Determine which observation pieces we want to include. 都是 one-hot的形式，所以才有这个维度
        # 决定一个玩家可以看的那些信息
        pieces = [("player", 2, (2,))]  # one-hot的形式表示当前是哪个玩家的回合
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:  # 私人的信息
            pieces.append(("private_card", 3, (3,)))  # （one-hot）（手中的牌）
        if iig_obs_type.public_info:  # 公共的信息
            if iig_obs_type.perfect_recall:  # one-hot的形式记录决策路径，最多有三个回合，所以需要三行，左边代表P，右边代表B
                pieces.append(("betting", 6, (3, 2)))
            else:  # 如果不允许回忆之前的路径，就看当前的资金池的信息
                pieces.append(("pot_contribution", 2, (2,)))

        # Build the single flat tensor. 看需要多少列才能表达了所有的信息
        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, np.float32)  # 那么一个tensor就包括想要的所有信息

        # Build the named & reshaped views of the bits of the flat tensor.
        # 重新搭建之前坦平的tensor
        self.dict = {}  # 定义一个字典来记录玩家看到的obs
        index = 0  # 记录一个索引，用来隔断各个信息
        for name, size, shape in pieces:  # 循环所有的obs，用字典的形式记录所有的信息
            self.dict[name] = self.tensor[index:index + size].reshape(shape)
            index += size

    def set_from(self, state, player):  # 该方式返回的是字典的形式
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        self.tensor.fill(0)  # 首先tensor中的所有值都是0
        if "player" in self.dict:
            self.dict["player"][player] = 1  # 用【0，1】【1，0】表示哪个玩家
        if "private_card" in self.dict and len(state.cards) > player:  # 玩家手里得有牌
            self.dict["private_card"][state.cards[player]] = 1  # 【1,0,0】[0,1,0][0,0,1] 分别表示哪张牌
        if "pot_contribution" in self.dict:  # 想要看到赌注
            self.dict["pot_contribution"][:] = state.pot  # 直接复制，不用one-hot
        if "betting" in self.dict:  # 看看有没有押注
            for turn, action in enumerate(state.bets):  # 返回（索引，值）
                self.dict["betting"][turn, action] = 1  # 返回的是进行的什么操作

    def string_from(self, state, player):  # 返回string的形式，更加符合人类的语言
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        if "player" in self.dict:
            pieces.append(f"p{player}")  # p0, p1
        if "private_card" in self.dict and len(state.cards) > player:
            pieces.append(f"card:{state.cards[player]}")  # card:0,card:1,card:2
        if "pot_contribution" in self.dict:
            pieces.append(f"pot[{int(state.pot[0])} {int(state.pot[1])}]")  # pot[2 2]
        if "betting" in self.dict and state.bets:
            pieces.append("".join("pb"[b] for b in state.bets))  # 'pb'
        return " ".join(str(p) for p in pieces)


pyspiel.register_game(_GAME_TYPE, KuhnPokerGame)  # 注册游戏，然后很多方法就可以直接使用了
