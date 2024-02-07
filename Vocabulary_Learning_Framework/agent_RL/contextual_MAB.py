"""
define observation space, action space, reward function
1：如果只考虑音标对应的所有字母的概率分布的KL散度，那么大概率会选择比较长的音标最为最需要联系的单词
2：接下来要考虑的是正确字母和错误字母如何使用
"""
import numpy as np
from scipy.stats import entropy
import pandas as pd

# the testing result of forgetting student
# what the observation should be?
# first, select 2 from 15 words


'''
tasks = {('dʒ æ k ʌ t', 'j a c k e t'): ('jabatd', [1, 1, 0, 0, 0, 0], 0.5, 0.333),
               ('k ɝ ɪ r', 'c a r e e r'): ('cxrxff', [1, 0, 1, 0, 0, 0], 0.333, 0.333),
               ('f ɛ r w ɛ l', 'f a r e w e l l'): ('fexynvot', [1, 0, 0, 0, 0, 0, 0, 0], 0.25, 0.125),
               ('p ɑ l ɪ ʃ', 'p o l i s h'): ('phfopi', [1, 0, 0, 0, 0, 0], 0.5, 0.167),
               ('h ʌ b ɪ tʃ u ʌ l', 'h a b i t u a l'): ('hhqzktal', [1, 0, 0, 0, 0, 0, 1, 1], 0.5, 0.375),
               ('b r i ð', 'b r e a t h e'): ('bmejblr', [1, 0, 1, 0, 0, 0, 0], 0.286, 0.286),
               ('t ɔ l', 't a l l'): ('txor', [1, 0, 0, 0], 0.25, 0.25),
               ('k ɑ n t r æ s t', 'c o n t r a s t'): ('cvantiyq', [1, 0, 0, 0, 0, 0, 0, 0], 0.375, 0.25),
               ('k ɔ r d', 'c o r d'): ('copb', [1, 1, 0, 0], 0.5, 0.5),
               ('h ɔ l', 'h a l l'): ('hhkl', [1, 0, 0, 1], 0.5, 0.5),
               ('s ʌ b ɝ b', 's u b u r b'): ('saqhkb', [1, 0, 0, 0, 0, 1], 0.333, 0.333),
               ('f r aɪ d i', 'f r i d a y'): ('flplep', [1, 0, 0, 0, 0, 0], 0.167, 0.167),
               ('l ɪ k ɝ', 'l i q u o r'): ('lmfsyi', [1, 0, 0, 0, 0, 0], 0.333, 0.167),
               ('b aʊ', 'b o w'): ('bpu', [1, 0, 0], 0.333, 0.333),
               ('f i b ʌ l', 'f e e b l e'): ('fendnb', [1, 1, 0, 0, 0, 0], 0.5, 0.333)}
'''

tasks = [('dʒ æ k ʌ t', 6, 0.5, 0.333), ('k ɝ ɪ r', 6, 0.333, 0.333), ('f ɛ r w ɛ l', 8, 0.25, 0.125),
         ('p ɑ l ɪ ʃ', 6, 0.5, 0.167),
         ('h ʌ b ɪ tʃ u ʌ l', 8, 0.5, 0.375), ('b r i ð', 7, 0.286, 0.286), ('t ɔ l', 4, 0.25, 0.25),
         ('k ɑ n t r æ s t', 8, 0.375, 0.25), ('k ɔ r d', 4, 0.5, 0.5), ('h ɔ l', 4, 0.5, 0.5),
         ('s ʌ b ɝ b', 6, 0.333, 0.333),
         ('f r aɪ d i', 6, 0.167, 0.167), ('l ɪ k ɝ', 6, 0.333, 0.167), ('b aʊ', 3, 0.333, 0.333),
         ('f i b ʌ l', 6, 0.5, 0.333)]


# definition: the maximum improvement of memory, the maximum entropy between two memory.
# 使用word embedding技术，将音标和字母向量化，再加入其他所有的信息


class MultiArmBandit:
    def __init__(self, n_arms, observation):
        self.n_arms = n_arms
        self.arm_counts = np.zeros(n_arms)  # the chosen number of each task
        self.arm_values = np.zeros(n_arms)  # the value of chosen task
        self.observation = observation  # only use the phonemes

    def select_arm(self, epsilon):
        """
        leverage the ε-epsilon
        :param epsilon: prob of exploration
        :return: the index of arms
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_arms)  # randomly select
        else:
            return np.argmax(self.arm_values)  # exploiting, choose the maximum value

    def reward_function(self, arm, excellent, forget):
        """the relative entropy of two prob distribution
           要考虑准确度，完整度，拼写长度，音标长度，
           标准用最简单的特征概括大多数的任务
        """
        # entropy is one of the criteria
        accuracy = []
        completeness = []
        position_phoneme = []
        total_entropy = 0
        current_observation = self.observation[arm][0].split(' ')
        for position, phoneme in enumerate(current_observation):
            position_phoneme.append(phoneme + '_' + str(position))
        # find the prob distribution of two memory table
        for i in position_phoneme:
            excellent_prob = excellent.loc[i].values
            forget_prob = forget.loc[i].values
            total_entropy += entropy(excellent_prob, forget_prob, base=2)

        # 计算一个相对准确度差，然后直接加起来
        for ob in self.observation:
            accuracy.append(ob[2])
        relative_accuracy = [max(accuracy) - x for x in accuracy]
        avg_accuracy = relative_accuracy[arm]/self.observation[arm][1]

        # 计算一个相对完整度差，然后直接加起来
        for ob in self.observation:
            completeness.append(ob[3])
        relative_completeness = [max(completeness) - x for x in completeness]
        avg_completeness = relative_completeness[arm] / self.observation[arm][1]

        reward = total_entropy / len(position_phoneme) + avg_accuracy + avg_completeness
        return reward

    def update(self, chosen_arm, reward):
        """
        update reward
        :param chosen_arm: the index of arms
        :param reward: reward
        """
        self.arm_counts[chosen_arm] += 1
        n = self.arm_counts[chosen_arm]

        # 使用增量更新公式更新估计值
        value = self.arm_values[chosen_arm]
        new_value = value + (reward - value) / n
        self.arm_values[chosen_arm] = new_value


if __name__ == '__main__':
    # the number of arms is equal to the number of tasks
    n_tasks = len(tasks)
    bandit = MultiArmBandit(n_tasks, tasks)
    training = 500  # the training number
    task_value = {}
    # train 1000 times
    excellent_memory_df = pd.read_excel('excellent_memory.xlsx', index_col=0, header=0)  # excellent students
    forget_memory_df = pd.read_excel('forgetting_memory.xlsx', index_col=0, header=0)  # excellent students
    # 分别对音标和letter进行独热编码
    # 获取行索引，并转换为列表
    phonemes = excellent_memory_df.index.tolist()
    # 获取列名，并转换为列表
    letters = excellent_memory_df.columns.tolist()
    # one hot encoding
    one_hot_phonetics = {phonetic: [int(i == phonetic) for i in phonemes] for phonetic in phonemes}
    one_hot_letters = {letter: [int(i == letter) for i in letters] for letter in letters}

    for _ in range(training):
        exploration_rate = 0.5  # set the prob of exploration
        selected_arm = bandit.select_arm(exploration_rate)  # select arm
        selected_arm_reward = bandit.reward_function(selected_arm, excellent_memory_df,
                                                     forget_memory_df)  # calculate the rewards
        bandit.update(selected_arm, selected_arm_reward)  # 更新所选臂的估计值

    # 输出每个臂被选择的次数和估计值
    for i in range(len(tasks)):
        print(f"{tasks[i]} 被选择了 {bandit.arm_counts[i]}次，价值为{bandit.arm_values[i]}")
        task_value[tasks[i]] = bandit.arm_values[i]
    # 按照价值大小排序，则排好的顺序就是应该记忆的顺序,这种判定方式容易选择长的词
    sorted_dict = dict(sorted(task_value.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)
