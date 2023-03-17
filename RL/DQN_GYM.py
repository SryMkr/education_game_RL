"""DQN的实际运用,需要自己调整参数"""


import os
import random
import numpy as np
import gym
import tensorflow as tf
import tensorlayer as tl


ALG_NAME = 'DQN'
ENV_ID = 'CartPole-v1'


# 写一个经验回放的函数
class ReplayBuffer:
    def __init__(self, capacity=10000):  # 容量的大小位10000
        self.capacity = capacity
        self.buffer = []  # 一个列表来存放数据
        self.position = 0  # 有一个position的指针，应该是要挑选数据
        self.batch_size = 1000  # 一个batch大小的数据位1000

    # 函数 push 就是把智能体与环境交互的到的信息添加到经验池中，我只保留10000条数据，再多的话就覆盖掉之前的数据
    def push(self, state, action, reward, next_state, done):  # 存放的数据，下一个state的目的是计算状态价值
        if len(self.buffer) < self.capacity:  # 如果数据的容量小于10000，
            self.buffer.append(None)  # 则列表中不添加任何数据
        self.buffer[self.position] = (state, action, reward, next_state, done)  # 是不是说一个buffer满了就遗弃了之前的数据
        self.position = int((self.position + 1) % self.capacity)  # （0-9999）

    # 使用sample从经验队列中随机挑选一个batch_size的数据，使用zip函数把每一条数据打包到一起：
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)  # 从一个列表中，随机选择设定大小的数据
        # zip(*)代表解压数据，就是按照对应的column解压为列表，，map是将前面的函数运用到后面的所有数据上
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # 将里面的元组数据转换位列表数据
        """
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        # 返回的是一个batch size大小的所有数据
        return state, action, reward, next_state, done


# 定义agent
class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]  # 4 选择了4个游戏参数作为特征 （4，0）
        self.action_dim = self.env.action_space.n  # 【0，1】

        # 构建基本的神经网络
        def create_model(input_state_shape):
            # 神经网络的输入层 第一个参数是张量的维度 一维向量，有4行，第二个参数是第一层名字
            input_layer = tl.layers.Input(input_state_shape)
            # 中间层的节点数，中间层的激活函数，以及中间层的输入
            layer_1 = tl.layers.Dense(n_units=32, act=tf.nn.relu)(input_layer)
            layer_2 = tl.layers.Dense(n_units=16, act=tf.nn.relu)(layer_1)
            # 定义输出层的维度，与动作空间的维度一致 一维的列向量
            output_layer = tl.layers.Dense(n_units=self.action_dim)(layer_2)
            return tl.models.Model(inputs=input_layer, outputs=output_layer)

        # 网络的输入层是一个一维的列向量【0，4】 一个是过去网络的输出结果
        self.model = create_model([None, self.state_dim])
        # 网络的目标网络是一个一维的列向量【0，4】 一个网络的更新目标
        self.target_model = create_model([None, self.state_dim])
        self.model.train()  # 训练模型
        self.target_model.eval()  # 将这个网络设置为优化目标，这是评估网络
        # 优化模型，开始的学习率设置为一个较大的值，然后根据次数的增多，动态的减小学习率，以实现效率和效果的兼得
        self.model_optim = self.target_model_optim = tf.optimizers.Adam(lr=1e-3)
        self.epsilon = 0.9  # 模糊因子
        self.buffer = ReplayBuffer()  # 将过去的记录保存下来，用来训练模型
        self.gamma = 0.9  # 折扣奖励

    """Copy q network to target q network"""
    def target_update(self):
        for weights, target_weights in zip(
                self.model.trainable_weights, self.target_model.trainable_weights):
            target_weights.assign(weights)

    #  为每一个状态选择一个动作，所以必须有输入state
    def choose_action(self, state):
        if np.random.uniform() > self.epsilon: # 10%的概率随机挑选动作
            # 从动作空间中随机选择一个动作
            return np.random.choice(self.action_dim)
        else:
            # 90%的可能选择动作价值最大的动作
            q_value = self.model(state[np.newaxis, :])[0]  # 先增加一个维度是为了符合神经网络的输入，在选择动作是为了找到最大的动作
            return np.argmax(q_value)

    # 更新网络参数
    def replay(self):
        for _ in range(10):
            # sample an experience tuple from the dataset(buffer)
            states, actions, rewards, next_states, done = self.buffer.sample()  # 一个batch大小的所有数据
            # compute the target value for the sample tuple
            target = self.target_model(states).numpy()  # 输出的是这个batch的状态所对应的动作值
            next_target = self.target_model(next_states)  # 输出是这个batch中下一个状态所对应的动作值，区别在于state不一样，不用遍历
            # targets [batch_size, action_dim]
            # Target represents the current fitting level
            # next_q_values [batch_size, action_dim]
            next_q_value = tf.reduce_max(next_target, axis=1)   # 整个batch中下一个状态对应的动作的最大值
            # 将整个batch的动作值进行更新
            target[range(self.buffer.batch_size), actions] = rewards + (1 - done) * self.gamma * next_q_value

            # 在这里一直更新的是model的模型
            with tf.GradientTape() as tape:
                q_pred = self.model(states)   # 纯估计的值
                loss = tf.losses.mean_squared_error(target, q_pred)  # 使用均方根误差评估损失
            grads = tape.gradient(loss, self.model.trainable_weights)  # 梯度
            self.model_optim.apply_gradients(zip(grads, self.model.trainable_weights))  # 根据梯度优化权重（对应的那个动作）

    # 测试模型效果
    def test_episode(self, test_episodes):
        for episode in range(test_episodes):
            state = self.env.reset()
            state = state[0]
            total_reward, done = 0, False
            while not done:
                action = self.model(np.array([state], dtype=np.float32))[0]
                action = np.argmax(action)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = next_state.astype(np.float32)
                total_reward += reward
                state = next_state
                self.env.render()
            print("Test {} | episode rewards is {}".format(episode, total_reward))

    # 训练模型
    def train(self, train_episodes=200):
        #  训练模型
        for episode in range(train_episodes):
            total_reward, done = 0, False  # 初始化两个参数
            state = self.env.reset()  # 当前的状态是float类型
            state = state[0]  # 返回的元组里面有两个参数，第一个参数才是状态
            while not done:  # 如果游戏没有结束
                action = self.choose_action(state)  # 随机选择一个动作
                next_state, reward, done, _, _ = self.env.step(action)  # 做了这个动作会返回下一个状态，奖励，以及游戏结束没有
                next_state = next_state.astype(np.float32)  # 将下一个状态也转化为浮点型数据
                self.buffer.push(state, action, reward, next_state, done)  # 将得到的数据存到buffer中
                total_reward += reward  # 计算获得总奖励
                state = next_state # 将当前的状态改为下一个状态
                # self.render()
            if len(self.buffer.buffer) > self.buffer.batch_size:  # 如果buffer的大小已经大于batch__size
                self.replay()  # 训练模型
                self.target_update()  # 将model训练好的权重给target_model
            print('EP{} EpisodeReward={}'.format(episode, total_reward))
            # if episode % 10 == 0:
            #     self.test_episode(2)
        self.saveModel()
        # 测试训练好的模型
        self.loadModel()
        self.test_episode(test_episodes=5)

    # 保存模型
    def saveModel(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'model.hdf5'), self.model)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'target_model.hdf5'), self.target_model)
        print('Saved weights.')

    # 加载模型
    def loadModel(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if os.path.exists(path):
            print('Load DQN Network parametets ...')
            tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'model.hdf5'), self.model)
            tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'target_model.hdf5'), self.target_model)
            print('Load weights!')
        else: print("No model file find, please train model first...")


if __name__ == '__main__':
    env = gym.make(ENV_ID, render_mode='human')  # 创建游戏环境
    agent = Agent(env)  # agent得输入
    agent.train(train_episodes=200)  # 玩游戏的次数
    env.close()  # 关闭游戏环境
