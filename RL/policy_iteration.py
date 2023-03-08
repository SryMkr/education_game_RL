"""policy iteration  在状态价值 收敛以后才更新策略"""


import gym
import time
import numpy as np


# 评估策略
def policy_evaluation(env, value_table, policy, gamma=0.9, threshold=1e-4):
    delta = 2 * threshold  # 设定一个delta，当所有state的绝对值的平均数小于threshold的时候，不再更新状态
    while delta > threshold:  # 当delta大于设定的阈值
        # 创建一个新的state value 表格
        new_value_table = np.zeros(env.observation_space.n)  # 此处不能用np.copy(value_table),因为更新V(s)用的是+=，所以若不置零则无限加和不收敛
        for observation in range(env.observation_space.n):  # 循环每一个observation
            # 1：从当前observation提取策略对应的action，指向某一个动作
            action = policy[observation]  # 可能会由于随机选择的动作不包含奖励=1的动作而得到无更新的new_value_table
            # 2.更新state value  0: LEFT,1: DOWN,2: RIGHT,3: UP
            for prob, next_state, reward, terminated in env.P[observation][action]:
                # print('状态', observation, '动作', action, '其他信息', prob, next_state, reward, terminated)
                new_value_table[observation] += prob * (reward + gamma * value_table[next_state])
        delta = sum(np.fabs(new_value_table - value_table))  # 求解绝对值（每一个state前后的差值的绝对值的和，要小于设定好的阈值）
        value_table = new_value_table
    return value_table


def policy_improvement(env, value_table, policy, gamma=0.9):
    while True:
        old_policy = np.copy(policy)  # 存储旧policy
        for observation in range(env.observation_space.n):  # 循环每一个观察到observation
            action_value = np.zeros(env.action_space.n)  # 重新创建动作空间
            for action in range(env.action_space.n):  # 循环每一个动作
                for prob, next_state, reward, done in env.P[observation][action]:
                    action_value[action] += prob * (reward + gamma * value_table[next_state])  # 计算每个动作动作的价值
            # 2.更新最优policy,最大动作价值的索引
            policy[observation] = np.argmax(action_value)
        # print(policy)
        if np.all(policy == old_policy): break
    return policy


# 更新策略的方式
def policy_iteration(env, iterations, gamma=0.9):
    env.reset()  # 重置环境，两个返回值（初始状态的observation，额外信息用于输出想输出的值）
    start = time.time()  # 获得游戏开始的时间，以毫秒计算
    # 初始化策略-随机策略 该函数的作用是从 动作空间[0,4)中挑选每个observation下的动作 例如 [0 3 3 1 0 1 3 1 3 3 1 3 1 0 1 2] 数字代表移动的方向
    policy = np.random.randint(low=0, high=env.action_space.n, size=env.observation_space.n)
    # 初始化observation value (初始化0)
    value_table = np.zeros(env.observation_space.n)
    for step in range(iterations):  # episode的次数
        old_policy = np.copy(policy)  # 复制一份原先的策略，用于判断policy是否稳定
        # 1.Policy Evaluation根据当前的策略来计算V(S)的值
        value_table = policy_evaluation(env, value_table, policy, gamma)
        # 2.Policy Improvement
        policy = policy_improvement(env, value_table, policy, gamma)
        # 3.判断终止条件
        if np.all(policy == old_policy):
            print('===== Policy Iteration ======\nTime Consumption: {}s\nIteration: {} steps\nOptimal Policy(gamma={}): '
                  '{}'.format(time.time() - start, step + 1, gamma, policy))
            break
    return value_table, policy


# timesteps超过150个步骤还没有结束游戏则强行终止此次探索
def play_game(env, policy, episodes=5, timesteps=150):
    for episode in range(episodes):
        observation, _ = env.reset()
        print(policy)
        for t in range(timesteps):
            action = policy[observation]
            observation, reward, terminated, _, _ = env.step(action)
            print(observation)
            if terminated:
                print(
                    "===== Episode {} finished ====== \n[Reward]: {} [Iteration]: {} steps".format(episode + 1, reward,
                                                                                                   t + 1))
                env.render()
                break


# 从gym注册的游戏中调用一个游戏，desc:自定义自己想要的游戏环境，map_name：游戏自带地图，is_slippery动作会不会滑动，render_mode展示结果的方式
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode=None)
# 下面的两种方式，分别是调用可选择的动作的个数以及state可能存在的个数
# print(env.action_space.n, env.observation_space.n)
# 0: LEFT,1: DOWN,2: RIGHT,3: UP
# 策略迭代 (环境，迭代次数，折扣奖励)
value_table, policy = policy_iteration(env, iterations=1, gamma=0.9)
print(policy)
# # 使用迭代计算得到的策略打游戏
# play_game(env, policy, episodes=3)
# env.close()
