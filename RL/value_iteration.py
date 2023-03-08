# 边更新V边更新policy

import numpy as np
import gym


def extract_policy(old_value_table, gamma=0.9):
    policy = np.zeros(env.observation_space.n)  # 为每个状态创建一个策略
    for observation in range(env.observation_space.n):
        # q_sa: 在状态 s 下的 所有动作价值
        action_value = np.zeros(env.action_space.n)  # 所有的动作
        for action in range(env.action_space.n):
            for prob, next_state, reward, terminated in env.P[observation][action]:
                action_value[action] += prob * (reward + gamma * old_value_table[next_state])  # 计算每个动作动作的价值
        policy[observation] = np.argmax(action_value)
    return policy


def value_iteration(env, iterations, gamma=0.9, threshold=1e-20):
    value_table = np.zeros(env.observation_space.n)  # 初始化state的值
    for i in range(iterations):
        old_value_table = np.copy(value_table)  # 复制一份value_table
        for observation in range(env.observation_space.n):  # 循环计算每一个state
            action_value = np.zeros(env.action_space.n)  # 用来保存每个动作的价值
            for action in range(env.action_space.n):  # 在任何一个state下 循环每一个动作
                for prob, next_state, reward, done in env.P[observation][action]:
                    action_value[action] += prob * (reward + gamma * old_value_table[next_state])  # 计算每个动作动作的价值
            value_table[observation] = max(action_value)
        delta = np.sum(np.fabs(value_table-old_value_table))  # 计算一个差值
        if delta <= threshold:
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    return value_table


# timesteps超过150个步骤还没有结束游戏则强行终止此次探索
def play_game(env, policy, episodes=5, timesteps=150):
    for episode in range(episodes):
        observation, _ = env.reset()
        for t in range(timesteps):
            action = int(policy[observation])
            observation, reward, terminated, _, _ = env.step(action)
            print(observation)
            if terminated:
                print(
                    "===== Episode {} finished ====== \n[Reward]: {} [Iteration]: {} steps".format(episode + 1, reward,
                                                                                                   t + 1))
                env.render()
                break


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')  # 读取游戏参数
optimal_v = value_iteration(env, iterations=100, gamma=0.9)  # 值迭代的方式 迭代的次数和奖励的折扣
print('optimal_v :', optimal_v)
policy = extract_policy(optimal_v, gamma=0.9)
print('policy:', policy)
# Value-iteration converged at iteration# 1373.
# policy:[1 2 1 0 1 0 1 0 2 1 1 0 0 2 2 0]
#        [1. 2. 1. 0. 2. 1. 1. 0. 0. 2. 2. 0. 0. 0. 0. 0.]
# # 使用迭代计算得到的策略打游戏
play_game(env, policy, episodes=3)
env.close()
