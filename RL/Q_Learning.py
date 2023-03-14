"""一个简单的小游戏学习Q_TABLE
注意观察表格是如何更新的，理解更新也是一个逐渐向前递归的过程
"""

import time
import numpy as np
import pandas as pd

N_STATES = 6   # 状态集合
ACTIONS = ["left", "right"]  # 动作集合
EPSILON = 0.9  # 有10%的概率随机选择动作，有90%的概率选择收益最大的动作
learning_rate = 0.1  # 学习率
GAMMA = 0.9  # 折扣奖励
MAX_EPISODES = 15  # 玩多少次游戏，拿到奖励算一次
FRESH_TIME = 0.3  # 游戏更新步骤的时间
TerminalFlag = "terminal"  # 游戏结束的标志


# 初始化Q_TABLE，记录每种状态每个动作的价值
def build_q_table(n_states, actions):
    return pd.DataFrame(
        np.zeros((n_states, len(actions)), dtype=float),  # 初始化Q_TABLE
        columns=actions  # 列名
    )


# 选择动作 10%的概率随机选择动作，有90%的概率选择收益最大的动作 (不需要对动作有计算准确概率)
def choose_action(state, q_table):
    state_table = q_table.loc[state, :]   # 取得当前这个状态可以做的所有动作的动作价值的值
    if (np.random.uniform() > EPSILON) or ((state_table == 0).all()):  # 刚开的时候或者10%的概率随机选择动作
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_table.idxmax()  # 有90%的概率选择收益最大的动作
    return action_name


# 做了动作环境的反馈 状态s做了动作a
def get_env_feedback(S, A):
    if A == "right":  # 如果选择了右边，（游戏偏向于让agent向右移动）
        if S == N_STATES - 2:  # 如果已经到了最后一个位置
            S_, R = TerminalFlag, 10  # 将状态切换为终止，奖励为1
        else:
            S_, R = S + 1, 0  # 将状态向右移动，奖励为 0
    else:  # 向左边移动
        S_, R = max(0, S - 1), 0  # 将状态向右移动，奖励为 0  状态不能小于0
    return S_, R


#  游戏环境 输入当前状态，一共玩多少次游戏，多少步吃到了奖励
def update_env(S, episode, step_counter):
    env_list = ["-"] * (N_STATES - 1) + ["T"]  # （_____T）
    if S == TerminalFlag:
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print(interaction)
        time.sleep(2)
    else:
        env_list[S] = '0'  # 0是这个agent
        interaction = ''.join(env_list)  # 输出游戏的新画面
        print(interaction)
        time.sleep(FRESH_TIME)


# 训练过程
def rl():
    q_table = build_q_table(N_STATES-1, ACTIONS)  # 初始化Q表
    for episode in range(MAX_EPISODES):  # 一次又一次的玩游戏
        step_counter = 0  # 每一轮拿到宝藏走了多少步
        S = 0  # 从状态0开始
        is_terminated = False  # 游戏还没有结束
        update_env(S, episode, step_counter)  # 初始化游戏环境
        while not is_terminated:  # 如果游戏还没结束
            A = choose_action(S, q_table)  # 选择一个动作
            S_, R = get_env_feedback(S, A)  # 得到下一个状态，以及对应的奖励
            q_predict = q_table.loc[S, A]  # Q表内q(s,a)的原始值

            if S_ != TerminalFlag:  # 如果游戏没有结束
                q_target = R + GAMMA * q_table.loc[S_, :].max()  # 这一步可以预期的价值
            else:  # 游戏结束
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += learning_rate * (q_target - q_predict)  # 更新Q表
            S = S_
            update_env(S, episode, step_counter + 1)
            step_counter += 1  # 主要是记录步数
            print(q_table)
    return q_table


if __name__ == '__main__':
    q_table = rl()





