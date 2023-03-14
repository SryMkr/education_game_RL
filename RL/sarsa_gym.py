"""sarsa代码展示，和Q基本一样只是多了一个后续动作，更新方式不一样"""

import gym
import numpy as np


class Sarsa:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.9, e_greed=0.9):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = e_greed
        self.Q_table = np.zeros((state_dim, action_dim))

    def sample(self, state):
        if np.random.uniform() > self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.predict(state)
        return action

    def predict(self, state):
        all_actions = self.Q_table[state, :]
        max_action = np.max(all_actions)
        max_action_list = np.where(all_actions == max_action)[0]
        action = np.random.choice(max_action_list)
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * self.Q_table[next_state, next_action]  # 根据下一个状态的动作更新动作价值
        self.Q_table[state, action] += self.lr * (target_q - self.Q_table[state, action])

    def save(self):
        npy_file = '/model/sarsa_q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def load(self, npy_file='/model/sarsa_q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


class Agent:
    def __init__(self, env):
        self.env = env
        self.lr = 0.1,
        self.gamma = 0.9,
        self.e_greed = 0.1
        self.model = Sarsa(env.observation_space.n, env.action_space.n)

    def train(self, max_episode):
        for episode in range(max_episode):
            ep_reward, ep_steps = self.run_episode(render=False)
            if episode % 20 == 0:
                print('Episode %03s: steps = %02s , reward = %.1f' % (episode, ep_steps, ep_reward))
        self.model.save()
        # 下面是测试训练好的效果
        self.model.load()
        self.test_episode(render=True)

    def run_episode(self, render=False):
        total_reward = 0
        total_steps = 0
        state = self.env.reset()
        state = state[0]
        action = self.model.sample(state)
        while True:
            next_state, reward, terminated, _, _ = self.env.step(action)
            next_action = self.model.sample(next_state)  # 这里要为下一个状态也选择一个动作
            # 训练 Q-learning算法
            self.model.learn(state, action, reward, next_state, next_action, terminated)

            state = next_state
            action = next_action
            total_reward += reward
            total_steps += 1
            if render: self.env.render()
            if terminated: break
        return total_reward, total_steps

    def test_episode(self, render=False):
        total_reward = 0
        actions = []
        state = self.env.reset()
        state = state[0]
        while True:
            action = self.model.predict(state)
            next_state, reward, terminated, _, _ = self.env.step(action)
            state = next_state
            total_reward += reward
            actions.append(action)
            if render: self.env.render()
            if terminated: break

        print('test reward = %.1f' % (total_reward))
        print('test action is: ', actions)


if __name__ == '__main__':
    # 使用gym创建迷宫环境，设置is_slippery为False降低环境难度
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')
    agent = Agent(env=env)
    agent.train(500)
