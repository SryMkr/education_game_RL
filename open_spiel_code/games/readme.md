这个文件里包含了4个游戏，这四个游戏是open-spiel/python/games里面的游戏，了解这四个游戏有助于了解如何注册一个open-spiel格式的游戏。以及对游戏中所有可能
涉及到的专有名词有一定的了解。接下来介绍一些专有名词，以及注册一个游戏的基本框架。

Step_1: 导入包，特别的一定包括import pyspiel,因为得继承里面的一些类实现自己的游戏，其他的包自己看情况使用。

Step_2: 定义动作，看几个玩家能采取的动作一直是一样的？还是随着游戏的动作跟着变化

Step_3: 一定要实现两个常量：

_GAME_TYPE = pyspiel.GameType(
    short_name="python_kuhn_poker",  # 环境名字
    long_name="Python Kuhn Poker",
    # 动态一共有三种类型：1：sequential 玩家有先后顺序 kuhn_poker 2: simultaneous game (石头剪刀布) 3：mean field game
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL, 
    # deterministic：没有机会玩家； EXPLICIT_STOCHASTIC 游戏中有机会玩家,概率已知；Sampled_STOCHASTIC,游戏中有机会玩家,概率未知
    # 这个东西在扑克牌的例子中就是美丽荷官在线发牌，在这个例子中概率是已知的
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    # one-shot这种类型没搞明白; PERFECT_INFORMATION，玩家观察到的信息一样，IMPERFECT_INFORMATION，玩家观察到的信息不一样，受到private information的影响（扑克牌游戏）
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,  # 不完全信息的游戏
    # identical表示玩家的回报是一样的，对称的（win +1, loss,-1, tie,0）
    # general_sum 所有玩家的收益总和并不确定，不限制与一样，或者常数，或者为0（囚徒困境）
    utility=pyspiel.GameType.Utility.ZERO_SUM,  # zero_sum 零和游戏, constant_sum,收益总和是一个常数
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,  # 返回rewards（在游戏中间返回收益），或者返回terminal（在游戏结束返回收益）
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
    min_utility=-2.0,  # 最小得分，就是一个游戏中的最高得分和最低得分
    max_utility=2.0,  # 最大得分
    utility_sum=0.0,  # 效用总和为0，因为是个零和游戏
    max_game_length=3)  # e.g. Pass, Bet, Bet # 最大的游戏步长
    
Step_4: 实现游戏类：里面要有三种方法是强制需要的
    class YOUR_GAME_NAME(pyspiel.Game):
      def __init__(self, params=None):
          super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
      # 实现state
      def new_initial_state(self):  # 返回游戏最开始的状态，并且可以通过返回的类调用state所有想知道的值
          """Returns a state corresponding to the start of a game."""
          return KuhnPokerState(self)
      # 实现observer,这块用自己的方法实现就行，没必要用他们的接口，具体查看test/observation.py
      def make_py_observer(self, iig_obs_type=None, params=None):
          """Returns an object used for observing game state."""
          return KuhnPokerObserver(iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params)
          
Step_5: 实现state类
    # 定义这个游戏的state可能存在的所有信息
    class KuhnPokerState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        # 一些自己用到的变量
        self._game_over = False  # 判断游戏是否结束，初始化的时候游戏肯定没结束
        self._next_player = 0
    
    # 如何切换玩家
    def current_player(self):
      
    # 玩家能够采取的动作
    def _legal_actions(self, player):  # 输入当前是哪个玩家，以及能够采取的动作
        """Returns a list of legal actions, sorted in ascending order."""
        assert player >= 0  # 用于判断一个表达式，结果为false的时候直接触发异常
        return 返回可采取的动作
    # 机会玩家怎么做
    def chance_outcomes(self):  # 机会玩家可以采取的动作，及其概率
        """Returns the possible chance outcomes and their probabilities."""
        返回（value, probability）
   
    # 做了动作以后，游戏需要做那些改变
    def _apply_action(self, action):  # 游戏中一共有三个玩家，要确定是哪个玩家，以及可以采取的动作及动作的概率
    # 转化为string类型
    def _action_to_string(self, player, action):  # 输入玩家和采取的动作
      

    def is_terminal(self):  # 游戏是否结束
        """Returns True if the game is over."""
        return self._game_over

    def returns(self):  # 返回一轮结束后的输赢奖励
        """Total reward for each player over the course of the game so far."""
        

    def __str__(self):  # 返回episode
        """String for debug purposes. No particular semantics are required."""
       
    
  Step_6: 实现observer类,这个可以只需要考虑三种类型的信息 public information, private information, perfect_recall, params,
        class KuhnPokerObserver:  # 游戏的观察者，可能是返回某一个observation
        def __init__(self, iig_obs_type, params):
          self.tensor = 
           self.dict = {}  # 定义一个字典来记录玩家看到的obs
        
        
        def set_from(self, state, player):  # 该方式返回的是字典的形式  
           pass
        def string_from(self, state, player):  # 返回string的形式，更加符合人类的语言
           pass
    
  Step_7: 注册游戏 pyspiel.register_game(_GAME_TYPE, YOUR_GAME)
    
    以上就是注册一个游戏的基本流程，具体的理解参考具体游戏，遇到参数一定要仔细的扣他们的不同点，有助于对open-spiel框架有一个系统的了解
           
           
