"""
本文件是open-spiel中，python版本的所有的核心API
自己取消注释测试方法 所有的方法都在open-spiel/doc文件中
"""

# import pyspiel  # 导入open-spiel的包
import numpy as np  # 导入numpy 包

# registered_names()返回注册在open-spiel中的游戏名字
# for short_name in pyspiel.registered_names():
#     print(short_name)

# --------------------以下介绍的是导入游戏的添加游戏参数的两种方式--------------------------------------
# game4 = pyspiel.load_game("goofspiel(imp_info=True,num_cards=5,points_order=descending)") # 圆括号方式
# game5 = pyspiel.load_game("goofspiel", {
#     "imp_info": True,
#     "num_cards": 5,
#     "points_order": "descending"
# })  # 字典方式

# --------------------以下介绍的是导入游戏的不添加游戏参数的两种方式--------------------------------------
# game = pyspiel.load_game("matrix_pd")   # Prisoner's dilemma
# state = game.new_initial_state()
# state.apply_actions([1, 1])  # 囚徒困境中运用动作的方式，SIMULTANEOUS游戏

# --------------------以下介绍的是imperfect game--------------------------------------
# game2 = pyspiel.load_game("kuhn_poker")  # 导入游戏
# state2 = game2.new_initial_state()  # 获得游戏的初始状态
# print(state2) # 随时可以查看当前的state
# print(state2.is_chance_node())  # 查看当前是不是机会玩家
# print(state2.current_player())  # 返回当前的玩家，机会玩家的ID是-1

# print(game2.information_state_tensor_shape()) # 表示一个状态的维度
# print(game2.information_state_tensor_size())  # 表示一个状态的维度有多大
# print(game2.max_chance_outcomes())  # 简单说就是机会玩家有多少不同可能的动作，没有的话就是0
# print(game2.max_game_length())  # 返回最大的游戏步长
# print(state2.chance_outcomes())  # 机会节点的动作，及其每个动作的概率
# state2.apply_action(0)  # 给当前玩家做动作
# outcomes = state2.chance_outcomes()  # 机会玩家的【action，probability】
# print(state2.chance_outcomes())
# action_list, prob_list = zip(*outcomes) # 解压动作
# action = np.random.choice(action_list, p=prob_list)
# state2.apply_action(action) # 运用动作
# state2.apply_action(0)  # 以下只要一直做动作就行，不需要知道当前的玩家是谁
# state2.apply_action(1)
# state2.apply_action(0)
# state2.apply_action(1)
# print(state2.information_state_string())  # 以string的形式就是返回当前玩家的决策
# print(state2.history())  # 打印的是代表状态的数字，并不知道做了什么
# print(state2.information_state_string(1))  # 以string的形式就是返回指定玩家的决策
# Player 0's turn.
# print(state2.information_state_tensor())  # 以tensor打印当前玩家的observation，注意tensor和string是如何相互表达的
# print(state.information_state_tensor(1))  # 以tensor打印指定玩家的observation


# --------------------以下介绍的是perfect game--------------------------------------
# game = pyspiel.load_game("tic_tac_toe")
# print(game.max_game_length())  # 获得游戏的最大步长
# print(game.min_utility())    # Output: -1，返回最小的收益
# print(game.max_utility())    # Output: 1 返回最大的收益
# print(game.num_distinct_actions())    # 返回游戏可以做的动作总量
# print(game.observation_tensor_shape())  # the size of each dimension
# print(game.observation_tensor_size())  # total number of values to represent observation tensor
# state1 = game.new_initial_state()  # 返回游戏对象，有时候可能是机会玩家则为空
# print(state1)
# print(state1.current_player())  # 其他基本上都是0，1组合代表两个玩家
# 游戏玩家自己切换了，不用非要获得当前的玩家，在state的函数中就已经实现了玩家的切换
# 创作bot不就是在应用动作这里么？，而且可以每做一次动作，就可以打印状态
# state1.apply_action(4)  # 有顺序的游戏
# print(game.action_to_string(0, 1))  # 与游戏状态无关，可能就是自己设定某个玩家的某个动作，基本上用不着
# state1.apply_action(1)
# print(state1.legal_actions())  # 获得当前玩家的合法动作
# print(state1.legal_actions(1))  # 获得指定玩家合法动作
# state1.apply_action(2)
# state1.apply_action(5)
# print(state1.rewards())  # 为了获得中间过程的奖励，只是对某个动作的奖励
# state1.apply_action(6)
# print(state1.returns())  # 返回累计的奖励，但是只是一局游戏的，如何累积？
# print(state1.rewards())  # 如果奖励在结尾则和returns差不多
# print(state1.observation_string())  # 查看当前的游戏状态
# print(state1.observation_string(1))  # 指定玩家的observation
# print(state1.observation_tensor())  # [整个盘面，一号玩家的操作，二号玩家的操作]
# print(state1.observation_tensor(0))

# print(state1.history())  # 打印所有玩家的游戏路径

# --------------------以下介绍的是打印游戏的状态，基本上和上面重复了--------------------------------------
# 下面的代码一般都组合使用，但是和其他的方法重复了，感觉没有必要。都是为了展示某一个时刻的一个游戏状态
# state_copy = game.deserialize_state(state.serialize()) # 这句就是为了获得游戏当前的状态

# --------------------以下介绍的是打印游戏的状态，基本上和上面重复了，一般都组合使用--------------------------------------
# serialized_data = pyspiel.serialize_game_and_state(game, state1)
# print(serialized_data)  # 会获得所有的游戏参数，包括游戏的名字，以及已经做的动作
# game_copy, state_copy = pyspiel.deserialize_game_and_state(serialized_data)
# print(game_copy)  # 获得游戏名字
# print(state_copy)  # 获得当前的游戏状态

