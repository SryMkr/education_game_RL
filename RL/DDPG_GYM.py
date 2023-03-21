"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.
Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/
Environment
-----------
Openai Gym Pendulum-v0, continual action space
Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-proactionsbility 0.6.0
tensorlayer >=2.0.0
To run
------
python tutorial_DDPG.py --train/test
"""

import os
import random
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl


ENV_ID = 'Pendulum-v1'  # environment id
RANDOM_SEED = 2  # random seed, can be either an int number or None

ALG_NAME = 'DDPG'
TRAIN_EPISODES = 200  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for training
MAX_STEPS = 200  # total number of steps for each episode

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement  更新目标策略网络采用软更新
MEMORY_CAPACITY = 5000  # size of replay buffer
BATCH_SIZE = 32  # update action batch size
VAR = 2  # control exploration，加一个噪声，增加探索


# 训练网络之前要收集足够的数据
class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity  # 收集多少数据开始训练网络
        self.buffer = []
        self.position = 0

    # 保存数据
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    # 选择所有的数据
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        # 返回所有数据的值
        return state, action, reward, next_state, done

    # 返回buffer的长度
    def __len__(self):
        return len(self.buffer)


class DDPG(object):
    def __init__(self, action_dim, state_dim, action_range, replay_buffer):
        self.replay_buffer = replay_buffer
        self.action_dim, self.state_dim, self.action_range = action_dim, state_dim, action_range
        self.var = VAR

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # 构建actor网络，输入为状态，输出一个动作
        def get_actor(input_state_shape, name=''):
            input_layer = tl.layers.Input(input_state_shape, name='A_input')
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(layer)
            # 输出值动作的空间，映射到【-1，1】
            layer = tl.layers.Dense(n_units=action_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(layer)
            layer = tl.layers.Lambda(lambda x: action_range * x)(layer)  # 将输出值映射到环境中的动作空间
            return tl.models.Model(inputs=input_layer, outputs=layer, name='Actor' + name)

        # 构建critic网络，对动作评分，需要状态和动作对其进行评分
        def get_critic(input_state_shape, input_action_shape, name=''):
            state_input = tl.layers.Input(input_state_shape, name='C_s_input')
            action_input = tl.layers.Input(input_action_shape, name='C_a_input')
            layer = tl.layers.Concat(1)([state_input, action_input])  # 合并为一个输入
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(layer)
            layer = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(layer)
            # 返回Q(S,A)的值
            return tl.models.Model(inputs=[state_input, action_input], outputs=layer, name='Critic' + name)

        self.actor = get_actor([None, state_dim])
        self.critic = get_critic([None, state_dim], [None, action_dim])
        self.actor.train()
        self.critic.train()

        # 复制权重
        def copy_para(from_model, to_model):
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor([None, state_dim], name='_target')  # 复制actor网络结构
        copy_para(self.actor, self.actor_target)  # 复制权重
        self.actor_target.eval()  # 作为评价网络

        self.critic_target = get_critic([None, state_dim], [None, action_dim], name='_target')  # 复制critic网络结构
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # 设置衰减因子

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    # 更新目标网络的权重，加权平均的方式
    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        paras = [self.actor.trainable_weights, self.critic.trainable_weights]  # 现在的权重
        self.ema.apply(paras)  #
        for i, j in zip([self.actor_target.trainable_weights, self.critic_target.trainable_weights], paras):
            i.assign(self.ema.average(j))

    def get_action(self, state, greedy=False):
        action = self.actor(np.array([state]))[0]
        if greedy:
            return action
        # 生成一个均值，标准差的正太分布，增加探索。正态分布的均值和方差不停的调整
        return np.clip(
            np.random.normal(action, self.var), -self.action_range, self.action_range
        ).astype(np.float32)  # add randomness to action selection for exploration

    def learn(self):
        """
        Update parameters
        :return: None
        """
        self.var *= .9995
        states, actions, rewards, states_, done = self.replay_buffer.sample(BATCH_SIZE)
        rewards = rewards[:, np.newaxis]
        done = done[:, np.newaxis]

        # 更新评价网络
        with tf.GradientTape() as tape:
            actions_ = self.actor_target(states_)  # 根据动作目标网络网络选择下一个状态的动作
            q_ = self.critic_target([states_, actions_])  # 根据评价网络计算下一个状态动作的价值
            target = rewards + (1 - done) * GAMMA * q_  # 计算真实的目标Q（s,a）值
            q_pred = self.critic([states, actions])  # 根据当前的评价网络计算Q(S,A)作为预估值
            td_error = tf.losses.mean_squared_error(target, q_pred)  # 计算目标与实际的均方根误差
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)  # 求梯度
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))  # 更新网络

        # 更新actor网络
        with tf.GradientTape() as tape:
            actions = self.actor(states)   # 输出动作
            q = self.critic([states, actions])  # 评价动作的价值
            actor_loss = -tf.reduce_mean(q)  # maximize the q，计算动作价值的平均值
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.ema_update()  # 复制权重

    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic_target.hdf5'), self.critic_target)

    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic_target.hdf5'), self.critic_target)


if __name__ == '__main__':
    env = gym.make(ENV_ID, render_mode='human').unwrapped

    # reproducible
    env.reset(seed=RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high  # scale action, [-action_range, action_range]

    buffer = ReplayBuffer(MEMORY_CAPACITY)
    agent = DDPG(action_dim, state_dim, action_range, buffer)

    t0 = time.time()
    # 训练模型
    all_episode_reward = []
    for episode in range(TRAIN_EPISODES):
        state = env.reset()[0].astype(np.float32)
        episode_reward = 0
        for step in range(MAX_STEPS):
            env.render()
            # Add exploration noise
            action = agent.get_action(state)
            state_, reward, done, info, _ = env.step(action)
            state_ = np.array(state_, dtype=np.float32)
            done = 1 if done is True else 0
            buffer.push(state, action, reward, state_, done)

            if len(buffer) >= MEMORY_CAPACITY:
                agent.learn()

            state = state_
            episode_reward += reward
            if done:
                break

        if episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
        print(
            'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                episode + 1, TRAIN_EPISODES, episode_reward,
                time.time() - t0
            )
        )
    agent.save()
    plt.plot(all_episode_reward)
    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    # test
    agent.load()
    for episode in range(TEST_EPISODES):
        state = env.reset()[0].astype(np.float32)
        episode_reward = 0
        for step in range(MAX_STEPS):
            env.render()
            state, reward, done, info,_ = env.step(agent.get_action(state, greedy=True))
            state = state.astype(np.float32)
            episode_reward += reward
            if done:
                break
        print(
            'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                episode + 1, TEST_EPISODES, episode_reward,
                time.time() - t0
            )
        )
    env.close()
