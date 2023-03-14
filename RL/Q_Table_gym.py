"""Q table的算法在游戏中的应用"""

import gym
import numpy as np


# Q_learning算法核心
class QLearning:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.9, e_greed=0.9):
        # 初始化非常重要的参数
        self.action_dim = action_dim  # 动作空间
        self.lr = lr  # 学习率
        self.gamma = gamma  # 奖励折扣
        self.epsilon = e_greed  # 增加explore的可能性
        self.Q_table = np.zeros((state_dim, action_dim))  # 创建一个Q_table: state_space * action_space

    # 根据状态选择动作，输入必须有状态
    def sample(self, state):
        if np.random.uniform() > self.epsilon:
            action = np.random.choice(self.action_dim)  # 10%的可能随机选择一个动作
        else:
            action = self.predict(state)  # 90%的可能选择动作价值最大的动作
        return action

    # 同样是选择动作，只不过可能最大的值有多个动作，还需要随机挑选一下，想的周到
    def predict(self, state):
        all_actions = self.Q_table[state, :]  # 得到该状态对应的动作价值
        max_action = np.max(all_actions)  # 找出最大的那个动作价值
        # 防止最大的 Q 值有多个，找出所有最大的 Q，然后再随机选择
        # where函数返回一个 array， 每个元素为下标
        max_action_list = np.where(all_actions == max_action)[0]  # 返回最大值的索引
        action = np.random.choice(max_action_list)  # 再在索引中随机挑选一个动作，最为当前的动作
        return action

    # 更新动作的价值
    def learn(self, state, action, reward, next_state, done):
        if done:  # 如果游戏结束，没有下一个状态了
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.Q_table[next_state, :])  # 将动作的最大奖励作为更新目标
        self.Q_table[state, action] += self.lr * (target_q - self.Q_table[state, action])

    # 保存 Q_table，训练结束用
    def save(self):
        npy_file = 'qlearning_table.npy'
        np.save(npy_file, self.Q_table)
        print(npy_file + ' saved.')

    # 读取 Q_table，测试效果用
    def load(self, npy_file='qlearning_table.npy'):
        self.Q_table = np.load(npy_file)
        print(npy_file + ' loaded.')


# 定义agent
class Agent:
    def __init__(self, env):
        self.env = env  # 交互的环境
        self.lr = 0.1  # 学习率
        self.gamma = 0.9  # 奖励折扣
        self.e_greed = 0.9  # 贪婪率
        self.model = QLearning(env.observation_space.n, env.action_space.n)  # Q_Learning的模型

    # 训练模型，输入最大的迭代次数
    def train(self, max_episode):
        for episode in range(max_episode):  # 一次一次的玩游戏
            ep_reward, ep_steps = self.run_episode(render=False)  # 一次游戏结束以后，得到的总奖励和总步长
            if episode % 20 == 0:
                print('Episode %03s: steps = %02s , reward = %.1f' % (episode, ep_steps, ep_reward))
        self.model.save()

        # 如果要测试学习结果需要将这块注释掉
        self.model.load()
        self.test_episode(render=True)

    # 在每一玩游戏中发生了什么
    def run_episode(self, render=False):
        total_reward = 0  # 总奖励
        total_steps = 0  # 总的步长
        state = self.env.reset()  # 重置环境,返回的是元组，第一个才是状态
        state = state[0] # state是一个元组，元组第一位才是状态
        while True:  # 一直循环，直到游戏结束
            action = self.model.sample(state)  # 根据当前的状态选择一个动作
            next_state, reward, terminated, _, _ = self.env.step(action)  # 在游戏中走一步，返回一些值
            # 根据返回的值 训练 Q-learning算法
            self.model.learn(state, action, reward, next_state, terminated)
            state = next_state  # 下一个状态为当前状态
            total_reward += reward  # 记录总的奖励数
            total_steps += 1  # 步长加一
            if render:  self.env.render()
            if terminated: break
        return total_reward, total_steps

    # 测试训练的结果如何
    def test_episode(self, render=False):
        total_reward = 0  # 总奖励
        actions = []  # 策略
        state = self.env.reset()  # 初始化游戏环境
        state = state[0]
        while True:
            action = self.model.predict(state)  # 选择最大的动作
            next_state, reward, terminated, _, _ = self.env.step(action)  # 得到一些返回值
            state = next_state
            total_reward += reward
            actions.append(action)
            if render: self.env.render()
            if terminated: break

        print('test reward = %.1f' % total_reward)
        print('test action is: ', actions)


if __name__ == '__main__':
    # 使用gym创建迷宫环境，设置is_slippery为False降低环境难度, render_mode可以设置为None
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')
    agent = Agent(env=env)  # 给agent放到一个环境中
    agent.train(200)  # 设置玩游戏的次数
