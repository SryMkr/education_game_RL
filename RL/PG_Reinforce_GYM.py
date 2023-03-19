"""
PG_reinforce，每一轮结束以后才训练模型
"""


import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl

ENV_ID = 'CartPole-v1'  # environment id
RANDOM_SEED = 1  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'PG'  # 算法的名字
TRAIN_EPISODES = 50  # 训练模型的次数
TEST_EPISODES = 10  # 测试的次数
MAX_STEPS = 500  # 每个episode的最大的游戏步长


# 策略迭代的模型
class PolicyGradient:
    # 初始化四个参数分别为 状态空间大小，动作空间大小，学习率，和奖励的折扣
    def __init__(self, state_dim, action_num, learning_rate=0.02, gamma=0.99):
        self.gamma = gamma  # 奖励的折扣
        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []  # 分别创建三个列表来存储数据

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)  # 以正太分布初始化权重
        b_init = tf.constant_initializer(0.1)  # 初始化偏差设置为常量
        input_layer = tl.layers.Input([None, state_dim], tf.float32)  # 网络的输入为样本量*状态的维度
        layer = tl.layers.Dense(n_units=30, act=tf.nn.tanh, W_init=W_init, b_init=b_init)(input_layer)  # 第一层
        all_act = tl.layers.Dense(n_units=action_num, act=None, W_init=W_init, b_init=b_init)(layer)  # 第二层
        self.model = tl.models.Model(inputs=input_layer, outputs=all_act)  # 网络模型
        self.model.train()  # 训练模型
        self.optimizer = tf.optimizers.Adam(learning_rate)  # 以adam的方式优化模型

    # 无论如何需要为网络选择动作，按照概率的方式随机抽样，输入一个状态，返回一个动作
    def get_action(self, s, greedy=False):
        _logits = self.model(np.array([s], np.float32))  # 输入一个状态以后会得到每个动作价值的输出值
        _probs = tf.nn.softmax(_logits).numpy()  # 将输出值转化为概率
        if greedy:
            # 如果采用贪婪算法，则返回最大值概率的动作
            return np.argmax(_probs.ravel())
        # _probs.ravel()将动作维度变为一维的，tl.rein是表示强化学习的模块，然后以一定的概率从一维动作空间中随机抽样
        return tl.rein.choice_action_by_probs(_probs.ravel())

    # 记录一个episode中所有的状态，与之对应的动作和奖励
    def store_transition(self, s, a, r):
        self.state_buffer.append(np.array([s], np.float32))
        self.action_buffer.append(a)
        self.reward_buffer.append(r)

    # 训练模型，更新权重
    def learn(self):
        discounted_reward = self._discount_and_norm_rewards()  # 获得每一个状态价值
        with tf.GradientTape() as tape:  # 用梯度下降的方法更新权重
            _logits = self.model(np.vstack(self.state_buffer))  # 预测所有的动作的概率
            # 计算交叉熵损失函数，前面一个是模型预测动作的概率，后面一个是标签（实际采取的动作），代表的是信息量的大小
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=_logits, labels=np.array(self.action_buffer))
            loss = tf.reduce_mean(neg_log_prob * discounted_reward)  # 各个动作概率的误差*该动作获得的期望一个episode奖励的平均值
            # loss = tl.rein.cross_entropy_reward_loss(
            #     logits=_logits, actions=np.array(self.action_buffer), rewards=discounted_reward)
        grad = tape.gradient(loss, self.model.trainable_weights)  # 计算梯度
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))  # 更新梯度
        # 将所有的数据清空，由此可得每个episode一次计算
        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []

    #  这玩意应该是为了计算在一个episode的情况下某一个状态下的状态价值
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_reward_buffer = np.zeros_like(self.reward_buffer)  # 创建与奖励维度一样的打折后的奖励
        running_add = 0  # 为了记录之后的奖励期望，相当于往前一直迭代,最后的输出结果是每一个状态对应一个状态价值
        for t in reversed(range(0, len(self.reward_buffer))):  # 从最后一个奖励开始倒着计算
            running_add = running_add * self.gamma + self.reward_buffer[t]
            discounted_reward_buffer[t] = running_add

        # 将所有的状态价值归一化处理
        discounted_reward_buffer -= np.mean(discounted_reward_buffer)
        discounted_reward_buffer /= np.std(discounted_reward_buffer)
        return discounted_reward_buffer

    # 保存训练好的模型 （权重）
    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):  # 如果文件夹不存在就创建文件夹
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'pg_policy.hdf5'), self.model)

    # 加载训练好的模型
    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))  # 权重保存的路径
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'pg_policy.hdf5'), self.model)  #将权重放到model中


if __name__ == '__main__':
    env = gym.make(ENV_ID, render_mode='human').unwrapped  # 不知道unwrapped是什么意思
    np.random.seed(RANDOM_SEED)  # 随机数种子的作用是为了让每次产生的随机数都尽量一样
    tf.random.set_seed(RANDOM_SEED)  # 给网络设置随机数种子，使得权重的初始值基本一样
    env.reset(seed=RANDOM_SEED)  # 给环境设置随机数种子，让环境的初始状态基本一样
    # 这里就是训练的过程，输入的参数为动作空间的大小，和state的维度
    agent = PolicyGradient(
        action_num=env.action_space.n,
        state_dim=env.observation_space.shape[0],
    )

    t0 = time.time()  # 记录时间
    # 训练模型
    all_episode_reward = []  # 记录每个episode的reward，这个是为了画图
    for episode in range(TRAIN_EPISODES):  # 训练的次数
        state = env.reset()[0]  # 初始化游戏状态,返回值有两个（state,information）
        episode_reward = 0  # 记录每一个episode获得的总奖励
        for step in range(MAX_STEPS):  # 一个episode中的最大步长
            if RENDER:  # 是否可视化训练过程
                env.render()
            action = agent.get_action(state)  # 采用随机抽样的方式选择动作，
            next_state, reward, done, _, _ = env.step(action)  # 输入动作，返回一些参数
            agent.store_transition(state, action, reward)  # 记录当前状态的状态，动作，奖励，是否结束，information
            state = next_state  # 转移到下一个状态
            episode_reward += reward  # 加总奖励
            if done:  # 游戏结束则停止循环
                break
        agent.learn()  # episode结束了，根据所有的获取的参数，更新权重
        print(
            'Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0
            )
        )
        # 第一次循环直接添加进去
        if episode == 0:
            all_episode_reward.append(episode_reward)
        # 后面的episode添加到列表的末尾
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

    env.close()  # 关闭游戏
    agent.save()  # 保存权重
    plt.plot(all_episode_reward)  # 画出获得奖励的图像
    if not os.path.exists('image'):  # 查看的文件地址是否存在
        os.makedirs('image')
    plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))  # 将图片保存

    # 测试训练好的模型，训练模型的时候注意注释下面的代码
    agent.load()  # 加载网络
    for episode in range(TEST_EPISODES):  # 需要测试网络的次数
        state = env.reset()[0]  # 初始化游戏状态,返回值有两个（state,information）
        episode_reward = 0  # 记录每一个episode获得的总奖励
        for step in range(MAX_STEPS):  # 每个episode最多走500次
            env.render()  # 可视化测试过程
            state, reward, done, info, _ = env.step(agent.get_action(state, True))  # 输入状态，取最大概率的动作，返回一些参数
            episode_reward += reward  # 累计获得的奖励
            if done:  # 如果游戏结束，则结束游戏
                break
        # 每轮游戏结束，输出episode，奖励以及游戏运行的时间
        print(
            'Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                episode + 1, TEST_EPISODES, episode_reward,
                time.time() - t0
            )
        )
    env.close()  # 测试结束关闭游戏
