"""
Actor-Critic
-------------
It uses TD-error as the Advantage.
To run
------
python tutorial_AC.py --train/test
"""

import time
import matplotlib.pyplot as plt
import os

import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl


ENV_ID = 'CartPole-v1'  # environment id
RANDOM_SEED = 2  # random seed, can be either an int number or None
# RENDER = False  # render while training

ALG_NAME = 'AC'
TRAIN_EPISODES = 200  # number of overall episodes for training
TEST_EPISODES = 10  # number of overall episodes for testing
MAX_STEPS = 500  # maximum time step in one episode
LAM = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic


#  创建actor的神经网络
class Actor(object):
    # 初始化一些参数，输入为state的维度和动作空间还有学习率
    def __init__(self, state_dim, action_dim, lr=0.001):
        input_layer = tl.layers.Input([None, state_dim])  # 神经网络的输入一直是状态的样本*状态的维度
        layer = tl.layers.Dense(n_units=32, act=tf.nn.relu6)(input_layer)  # 中间层的神经元数，激活函数，输入
        layer = tl.layers.Dense(n_units=action_dim)(layer)  # 输出层为动作空间的维度，以及输入
        self.model = tl.models.Model(inputs=input_layer, outputs=layer)  # 建立整个模型
        self.model.train()  # 训练模型
        self.optimizer = tf.optimizers.Adam(lr)  # 使用adam优化器，输入是学习率

    # 学习的过程就是更新权重的过程，输入为状态，动作，error
    def learn(self, state, action, td_error):
        with tf.GradientTape() as tape:  # 使用梯度下降的算法
            _logits = self.model(np.array([state]))  # 输出这个状态下，可采取动作的价值
            # 为什么还使用交叉熵损失函数
            _exp_v = tl.rein.cross_entropy_reward_loss(
                logits=_logits, actions=[action], rewards=td_error)
        grad = tape.gradient(_exp_v, self.model.trainable_weights)  # 根据损失做梯度下降
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))  # 更新权重

    # 根据状态选择一个动作，所以输入是状态
    def get_action(self, state, greedy=False):
        _logits = self.model(np.array([state]))  # 输出为所有动作的价值
        _prob = tf.nn.softmax(_logits).numpy()  # 将动作的价值转换为概率
        if greedy:
            # 对于确定策略，选择概率最大的动作
            return np.argmax(_prob.ravel())
        # 否则在所有的动作空间根据概率随机抽验
        return tl.rein.choice_action_by_probs(_prob.ravel())


# 创建一个critic的神经网络
class Critic(object):
    # 初始化一些参数，输入为状态空间，学习率
    def __init__(self, state_dim, lr=0.01):
        input_layer = tl.layers.Input([None, state_dim], name='state')  # 神经网络的输入一直是状态的样本*状态的维度
        layer = tl.layers.Dense(n_units=32, act=tf.nn.relu)(input_layer) # 中间层的节点数和激活函数，输入
        layer = tl.layers.Dense(n_units=1, act=None)(layer)  # 输出层只有一个，没有激活函数，只有一个原因是取的平均值
        self.model = tl.models.Model(inputs=input_layer, outputs=layer, name='Critic')  # 建立模型
        self.model.train()  # 训练模型
        self.optimizer = tf.optimizers.Adam(lr)  # 优化器输入为学习率

    # 更新critic的权重
    def learn(self, state, reward, state_, done):
        d = 0 if done else 1  # 观察游戏有没有结束
        with tf.GradientTape() as tape:  # 梯度下降法更新
            v = self.model(np.array([state]))  # 输入当前状态，输出当前动作价值（只有一个平均值）
            v_ = self.model(np.array([state_]))  # 输入下一个状态，输出下一个动作价值（只有一个平均值）
            td_error = reward + d * LAM * v_ - v  # 计算error error=reward+折扣*下一个状态价值的平均值-当前状态的平均值
            loss = tf.square(td_error)  # 将error取平方
        grads = tape.gradient(loss, self.model.trainable_weights)  # 计算梯度
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))  # 根据梯度更新权重
        # 返回误差
        return td_error


# 创建agent类
class Agent:
    # 初始化一些参数，输入为环境
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]  # 状态的维度为4
        self.action_dim = env.action_space.n  # 动作空间有两个
        self.actor = Actor(self.state_dim, self.action_dim, lr=LR_A)  # 创建actor的神经网络
        self.critic = Critic(self.state_dim, lr=LR_C)  # 创建critic的神经网络

    #  训练模型
    def train(self):
        # 训练模型
        self.train_episode()
        # 测试模型
        self.load()
        self.test_episode()

    #  训练模型
    def train_episode(self):
        t0 = time.time()  # 记录时间
        all_episode_reward = []  # 记录每一个episode的奖励，是为了画图用
        for episode in range(TRAIN_EPISODES):  # 玩游戏的次数
            state = env.reset()[0].astype(np.float32)  # 获得游戏的初始状态
            step = 0  # 记录游戏的step
            episode_reward = 0  # 记录每一轮奖励
            while True:
                env.render()  # 可视化训练过程
                action = self.actor.get_action(state)  # 选择一个动作
                state_, reward, done, _, _ = env.step(action)  # 做动作并返回一些参数
                state_ = state_.astype(np.float32)  # 将状态转换为浮点型
                episode_reward += reward  # 将episode的状态累加
                td_error = self.critic.learn(state, reward, state_, done)  # 计算动作的误差
                self.actor.learn(state, action, td_error)  # 根据误差更新actor网络的权重
                state = state_
                step += 1
                if done or step >= MAX_STEPS:
                    break
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

            print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))
            # Early Stopping for quick check
            if step >= MAX_STEPS:
                print("Early Stopping")  # Hao Dong: it is important for this task
                self.save()
                break
        # env.close()

        plt.plot(all_episode_reward)  # 画出所有奖励的图像
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))  # 保存图片

    #  测试训练好的模型
    def test_episode(self):
        t0 = time.time() # 开始时间
        for episode in range(TEST_EPISODES):  # 看看想要测试几次
            state = env.reset()[0].astype(np.float32)  # 获得初始状态
            t = 0  # number of step in this episode  # 计算一个episode中的步长
            episode_reward = 0  # 计算一个episode的总奖励
            while True:
                env.render()  # 可视化训练过程
                action = self.actor.get_action(state, greedy=True)  # 采用贪婪策略选择动作
                state_new, reward, done, info, _ = env.step(action)  # 走一步，返回一些参数
                state_new = state_new.astype(np.float32)  # 记录当前的状态
                if done:
                    reward = -20  # 如果游戏结束，给一个惩罚
                episode_reward += reward  # 记录游戏的总奖励
                state = state_new  # 将下一个状态变为当前状态
                t += 1  # 将游戏步长加1

                if done or t >= MAX_STEPS:  # 游戏结束或者超过最大的步长
                    # 输出一些信息
                    print('Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                          .format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))
                    break
        env.close()  # 关闭游戏环境

    # 保存训练好的权重,保存模型出现了问题
    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        # 原因是因为在每层的权重的维度不同
        tl.files.save_npz_dict(save_list=self.actor.model.all_weights, name=os.path.join(path, 'model_actor.npz'))
        tl.files.save_npz_dict(save_list=self.critic.model.all_weights, name=os.path.join(path, 'model_critic.npz'))
        print('Succeed to save model weights')

    # 加载训练好的权重和偏差
    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_critic.npz'), network=self.critic.model)
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_actor.npz'), network=self.actor.model)
        print('Succeed to load model weights')


if __name__ == '__main__':
    env = gym.make(ENV_ID, render_mode='human').unwrapped  # 读取游戏环境
    np.random.seed(RANDOM_SEED)  # 随机数种子
    env.reset(seed=RANDOM_SEED)  # 使得环境初始化的状态一样
    tf.random.set_seed(RANDOM_SEED)  # 使得权重和方差初始化的状态一样
    agent = Agent(env)  # 实例化agent
    agent.train()  # 训练agent
