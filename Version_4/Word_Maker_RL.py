# 先看一下有多少种组合的可能
import numpy as np
from itertools import product, chain
import pandas as pd

# # 以下代码是得出玩家玩游戏可能出现的所有可能的表现
#
# chances = [0, 1, 2, 3]  # 代表玩家使用了几次机会
# attempts = [1, 2, 3]  # 代表游戏的机会
# time = [30, 40, 60]  # 每种难度对应的时间
# time_difficulty = [[30, 3], [30, 4], [40, 2], [60, 1]]  # 关卡的难度
# # word_length = [3, 4, 5, 6, 7, 8]  # 单词长度 因为可能与时间有线性关系 所以不再考虑
# combination = np.zeros((1, 1), dtype=int)
# # (玩家用了几次机会，答题时长，当前任务难度，单词长度)
# for chance in chances:  # 循环玩家的表现次数
#     # 第一种情况，玩家使用了0次机会，说明了时间结束，没有回答任何问题，(奖励必然是错误的) 只有4种可能
#     if chance == 0:
#         chance_0 = [chances[0]]
#         chance_0 = product(chance_0, time_difficulty)  # 首先与时间组合
#         chance_0 = list(map(list, chance_0))
#         chance_0_list = []
#         for row in chance_0:  # [0, [30, 3], 3]
#             a = []
#             for i in row:
#                 if i == 0:
#                     a.append(i)
#                 else:
#                     for j in range(len(i)):
#                         a.append(i[j])
#             chance_0_list.append(a)
#
#     if chance == 1:  # 如果玩家使用了一次机会
#         #  如果当前机会是一次 那可对可错 1*30*2 = 60
#         chance_1_1 = [chances[1]]
#         chance_1_1 = product(chance_1_1, range(1, time[0] + 1, 1), [3, 4])  # 首先与时间组合
#         chance_1_1_list = list(map(list, chance_1_1))
#
#         # 如果当前机会是两次， 时间小于40秒是对，时间等于40秒是错 1*40*1 = 40
#         chance_1_2 = [chances[1]]
#         chance_1_2 = product(chance_1_2, range(1, time[1] + 1, 1), [2])  # 首先与时间组合
#         chance_1_2_list = list(map(list, chance_1_2))
#
#         # 如果当前机会是三次， 时间小于60秒是对，时间等于60秒是错 1*60*1 = 60
#         chance_1_3 = [chances[1]]
#         chance_1_3 = product(chance_1_3, range(1, time[2] + 1, 1), [1])  # 首先与时间组合
#         chance_1_3_list = list(map(list, chance_1_3))
#     if chance == 2:  # 如果玩家使用了两次机会
#         # 如果当前机会是两次， 那可对可错 1*40*1 = 40  如果是两次机会，那么难度只能是2
#         chance_2_2 = [chances[2]]
#         chance_2_2 = product(chance_2_2, range(1, time[1] + 1, 1), [2])  # 首先与时间组合
#         chance_2_2_list = list(map(list, chance_2_2))
#
#         # 如果当前机会是三次， 时间小于60秒是对，时间等于60秒是错 1*60*1 = 60 既然存在三次机会难度必然是1
#         chance_2_3 = [chances[2]]
#         chance_2_3 = product(chance_2_3, range(1, time[2]+1, 1), [1])  # 首先与时间组合
#         chance_2_3_list = list(map(list, chance_2_3))
#
#     if chance == 3:  # 如果玩家使用了三次机会
#         # 当前机会是三次， 那可对可错 时间不可能是0秒 1*60*1 = 60 时间是60秒得难度只能是1
#         chance_3 = [chances[3]]
#         chance_3 = product(chance_3, range(1, time[2]+1, 1), [1])  # 首先与时间组合
#         chance_3_list = list(map(list, chance_3))
#         print(chance_3_list)
#         print(len(chance_3_list))
# all_states = list(np.vstack((np.array(chance_0_list),np.array(chance_1_1_list),np.array(chance_1_2_list),
#                       np.array(chance_1_3_list),np.array(chance_2_2_list),np.array(chance_2_3_list),np.array(chance_3_list))))
#
# df = pd.DataFrame(all_states, columns=('attempts', 'time', 'difficulty'))
# # print(df)
# df.to_excel('saved_files/all_states.xls', sheet_name='states', header=True, index=False)

# 接下来要实现随机抽一条，作为玩家的表现
# 读取文件中的所有可能的玩家操作，并按照关卡分类

df = pd.read_excel('saved_files/all_states.xls', sheet_name='states')  # 读取可能的所有操作
# 创建每一种难度可能出现的操作
difficulty_1_action = []
difficulty_2_action = []
difficulty_3_action = []
difficulty_4_action = []
for value in df.values:  # 逐个打印某一行
    if value[2] == 1:  # 如果当前的难度为1
        difficulty_1_action.append(list(value))
    elif value[2] == 2:  # 如果当前的难度为1
        difficulty_2_action.append(list(value))
    elif value[2] == 3:  # 如果当前的难度为1
        difficulty_3_action.append(list(value))
    elif value[2] == 4:  # 如果当前的难度为1
        difficulty_4_action.append(list(value))
# 每一种难度允许出现的操作
difficulty_1_action = pd.DataFrame(difficulty_1_action, columns=('attempts', 'time', 'difficulty'))
difficulty_2_action = pd.DataFrame(difficulty_2_action, columns=('attempts', 'time', 'difficulty'))
difficulty_3_action = pd.DataFrame(difficulty_3_action, columns=('attempts', 'time', 'difficulty'))
difficulty_4_action = pd.DataFrame(difficulty_4_action, columns=('attempts', 'time', 'difficulty'))
difficulty_4_action.to_excel('saved_files/difficulty_4_action.xls', sheet_name='states', header=True, index=False)
difficulty_3_action.to_excel('saved_files/difficulty_3_action.xls', sheet_name='states', header=True, index=False)
difficulty_2_action.to_excel('saved_files/difficulty_2_action.xls', sheet_name='states', header=True, index=False)
difficulty_1_action.to_excel('saved_files/difficulty_1_action.xls', sheet_name='states', header=True, index=False)