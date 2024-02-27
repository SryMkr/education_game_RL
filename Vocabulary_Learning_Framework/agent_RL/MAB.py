"""
define observation space, action space, reward function
1：如果只考虑音标对应的所有字母的概率分布的平均KL散度，the larger the kl 散度, the larger the memory difference.
2: 结论好像是平均KL散度小的的先训练，反而能够快速提高准确度？
2：接下来要考虑的是正确字母和错误字母如何使用
3：第一个实验，只考虑错误的，计算错误字母的散度，纠正是全部纠正
KeyError: "['e_0'] not in index"这玩意到底是哪出错了？我应该有全部的索引
"""
import os
import Levenshtein
import string
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import pandas as pd
import pickle
import random

from Spelling_Framework.utils.choose_vocab_book import ReadVocabBook

with open('agent_RL/history_information.pkl', 'rb') as pkl_file:
    tasks = pickle.load(pkl_file)

selected_items = random.sample(tasks.items(), 50)

CURRENT_PATH = os.getcwd()  # get the current path
VOCAB_PATH: str = os.path.join(CURRENT_PATH, 'vocabulary_books', 'CET4', 'newVocab.json')  # get the vocab data path

corpus_instance = ReadVocabBook(vocab_book_path=VOCAB_PATH,
                                vocab_book_name='CET4',
                                chinese_setting=False,
                                phonetic_setting=True,
                                POS_setting=False,
                                english_setting=True)
original_corpus = corpus_instance.read_vocab_book()
random.shuffle(original_corpus)  # [['p ɑ p j ʌ l eɪ ʃ ʌ n', 'p o p u l a t i o n'], ['n aɪ n t i n', 'n i n e t e e n']


def add_position(corpus):
    """add position for each corpus"""
    corpus_with_position = []
    for pair in corpus:
        phonemes_position = ''
        letters_position = ''
        pair_position = []
        phonemes_list = pair[0].split(' ')
        for index, phoneme in enumerate(phonemes_list):
            phoneme_index = phoneme + '_' + str(index)
            phonemes_position = phonemes_position + phoneme_index + ' '
        letters_list = pair[1].split(' ')

        for index, letter in enumerate(letters_list):
            letter_index = letter + '_' + str(index)
            letters_position = letters_position + letter_index + ' '
        pair_position.append(phonemes_position.strip())
        pair_position.append(letters_position.strip())
        corpus_with_position.append(pair_position)
    return corpus_with_position


t = add_position(original_corpus)


class MultiArmBandit:
    def __init__(self, n_arms, observation):
        self.n_arms = n_arms
        self.arm_counts = np.zeros(n_arms)  # the chosen number of each task
        self.arm_values = np.zeros(n_arms)  # the value of chosen task
        self.observation = observation  # only use the phonemes
        self.accuracy = np.zeros(n_arms)

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
        # # entropy is one of the criteria
        # accuracy = []
        completeness = []
        #
        # # method 1； whole KL entropy
        # position_phoneme = []
        # total_entropy = 0
        # print(self.observation)
        # current_observation = self.observation[arm][0][0].split(' ')
        # for position, phoneme in enumerate(current_observation):
        #     position_phoneme.append(phoneme + '_' + str(position))
        # # find the prob distribution of two memory table
        # for i in position_phoneme:
        #     excellent_prob = excellent.loc[i].values
        #     forget_prob = forget.loc[i].values
        #     total_entropy += entropy(excellent_prob, forget_prob, base=2)
        # # reward = total_entropy / len(position_phoneme)

        # method 1； whole KL entropy
        position_phoneme = []
        total_entropy = 0

        current_observation = self.observation[arm][0][0].split(' ')
        for position, phoneme in enumerate(current_observation):
            position_phoneme.append(phoneme + '_' + str(position))
        # 找到字典中value为0的索引，并构造26个字母和索引，作为列索引
        zero_indices = [index for index, value in enumerate(self.observation[arm][1][0][0].values()) if value == 0]
        if len(zero_indices):
            for index in zero_indices:
                columns = [letter + '_' + str(index) for letter in string.ascii_lowercase]
                # find the prob distribution of two memory table
                for i in position_phoneme:
                    excellent_prob = excellent.loc[i, columns].values
                    forget_prob = forget.loc[i, columns].values
                    total_entropy += entropy(excellent_prob, forget_prob, base=2)
            reward = total_entropy / len(zero_indices)
        else:
            for i in position_phoneme:
                excellent_prob = excellent.loc[i].values
                forget_prob = forget.loc[i].values
                total_entropy += entropy(excellent_prob, forget_prob, base=2)
            reward = total_entropy / len(position_phoneme)
        accuracy = self.observation[arm][1][0][1]
        # print(accuracy)
        # 计算一个相对准确度差，然后直接加起来
        # for ob in self.observation:
        #     accuracy.append(ob[1][0][1])
        # relative_accuracy = [max(accuracy) - x for x in accuracy]
        # avg_accuracy = relative_accuracy[arm] / len(self.observation[arm][1][0][0])

        # 计算一个相对准确度差，然后直接加起来
        # for ob in self.observation:
        #     accuracy.append(ob[1][0][1])
        # relative_accuracy = [max(accuracy) - x for x in accuracy]
        # avg_accuracy = relative_accuracy[arm] / len(self.observation[arm][1][0][0])
        #
        # # 计算一个相对完整度差，然后直接加起来
        # for ob in self.observation:
        #     completeness.append(ob[1][0][2])
        # relative_completeness = [max(completeness) - x for x in completeness]
        # avg_completeness = relative_completeness[arm] / len(self.observation[arm][1][0][0])
        # reward = total_entropy / len(position_phoneme) + avg_accuracy + avg_completeness
        return reward, accuracy

    def update(self, chosen_arm, reward, accuracy):
        """
        update reward
        :param chosen_arm: the index of arms
        :param reward: reward
        """
        self.arm_counts[chosen_arm] += 1
        n = self.arm_counts[chosen_arm]
        self.accuracy[chosen_arm] = accuracy
        # 使用增量更新公式更新估计值
        value = self.arm_values[chosen_arm]
        new_value = value + (reward - value) / n
        self.arm_values[chosen_arm] = new_value


class evaluate_improvement:
    def __init__(self, memory, corpus):
        self.memory = memory
        self.corpus = corpus
        self.student_answer_pair = []
        self.accuracy = []
        self.completeness = []
        self.perfect = []
        self.avg_accuracy = []
        self.avg_completeness = []
        self.avg_perfect = []

    def generate_answer(self):
        """ generate answer based on the given phonemes,而且我要知道答案的长度，然后根据所有的音标对每一个位置选择最大值"""
        for phonemes, answer in self.corpus:
            phonemes = phonemes.split(' ')
            answer = answer.split(' ')
            spelling = []
            answer_length = len(answer)
            alphabet = string.ascii_lowercase
            for i in range(answer_length):
                # 将26个字母和位置结合起来，组成列索引
                if i == 0:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.memory.loc[phonemes[0], result_columns]
                    letter = possible_results.idxmax()
                else:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.memory.loc[phonemes, result_columns]
                    letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                    letter = letters_prob.idxmax()
                spelling.append(letter)
            self.student_answer_pair.append([spelling, answer])

    def evaluation(self):
        for stu_answer, correct_answer in self.student_answer_pair:
            stu_answer = ''.join([i.split('_')[0] for i in stu_answer])
            correct_answer = ''.join([i.split('_')[0] for i in correct_answer])
            word_accuracy = round(Levenshtein.ratio(correct_answer, stu_answer), 2)
            word_completeness = round(1 - Levenshtein.distance(correct_answer, stu_answer) / len(correct_answer), 2)
            word_perfect = 0.0
            if stu_answer == correct_answer:
                word_perfect = 1.0
            self.accuracy.append(word_accuracy)
            self.completeness.append(word_completeness)
            self.perfect.append(word_perfect)
        self.avg_accuracy = sum(self.accuracy) / len(self.accuracy)
        self.avg_completeness = sum(self.completeness) / len(self.completeness)
        self.avg_perfect = sum(self.perfect) / len(self.perfect)
        return self.avg_accuracy, self.avg_completeness, self.avg_perfect


if __name__ == '__main__':
    # the number of arms is equal to the number of tasks
    n_tasks = len(selected_items)
    bandit = MultiArmBandit(n_tasks, selected_items)
    epoch = 200  # the training number
    task_value = {}
    # train 1000 times
    excellent_memory_df = pd.read_excel('agent_RL/excellent_memory.xlsx', index_col=0, header=0)  # excellent students
    forget_memory_df = pd.read_excel('agent_RL/forgetting_memory.xlsx', index_col=0, header=0)  # forget students
    # # 分别对音标和letter进行独热编码
    # # 获取行索引，并转换为列表
    # phonemes = excellent_memory_df.index.tolist()
    # # 获取列名，并转换为列表
    # letters = excellent_memory_df.columns.tolist()
    # # one hot encoding
    # one_hot_phonetics = {phonetic: [int(i == phonetic) for i in phonemes] for phonetic in phonemes}
    # one_hot_letters = {letter: [int(i == letter) for i in letters] for letter in letters}

    for _ in range(epoch):
        exploration_rate = 0.5  # set the prob of exploration
        selected_arm = bandit.select_arm(exploration_rate)  # select arm
        selected_arm_reward, selected_arm_accuracy = bandit.reward_function(selected_arm, excellent_memory_df,
                                                                            forget_memory_df)  # calculate the rewards
        bandit.update(selected_arm, selected_arm_reward, selected_arm_accuracy)  # 更新所选臂的估计值

    # 输出每个臂被选择的次数和估计值
    for i in range(len(selected_items)):
        task_value[selected_items[i][0]] = (bandit.arm_values[i], bandit.accuracy[i])
    # 按照价值大小排序，则排好的顺序就是应该记忆的顺序,这种判定方式容易选择长的词
    sorted_dict = dict(sorted(task_value.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)
    # 评估选择的单词好不好的方式是，选择的单词是不是对整体记忆影响最大的
    # 要做一个拼接，将好的库里面对应的行，替换掉坏的里面对应的行，行成一个新的表格然后评估
    excellent_acc_list = []
    excellent_com_list = []
    excellent_per_list = []
    forget_acc_list = []
    forget_com_list = []
    forget_per_list = []
    excellent = evaluate_improvement(excellent_memory_df, t)
    excellent.generate_answer()
    excellent_acc, excellent_com, excellent_per = excellent.evaluation()
    print('excellent accuracy', excellent_acc)
    excellent_acc_list.append(excellent_acc)
    excellent_com_list.append(excellent_com)
    excellent_per_list.append(excellent_com)

    forget = evaluate_improvement(forget_memory_df, t)
    forget.generate_answer()
    forget_acc, forget_com, forget_per = forget.evaluation()
    print('forget_acc', forget_acc)
    forget_acc_list.append(forget_acc)
    forget_com_list.append(forget_com)
    forget_per_list.append(forget_com)
    n = len(sorted_dict)
    improvement_acc_list = []
    improvement_com_list = []
    improvement_per_list = []
    accuracy_list = []
    kl_divergence = []
    for key, value in sorted_dict.items():
        position_phoneme = []
        for position, phoneme in enumerate(key[0][0].split(' ')):
            position_phoneme.append(phoneme + '_' + str(position))
        # forget_memory_copy = forget_memory_df.copy()
        # 将 df2 中指定行的值赋值给 df1 的副本中相应的行
        try:
            forget_memory_df.loc[position_phoneme] += 0.8 * excellent_memory_df.loc[position_phoneme].values
        except KeyError as e:
            print(position_phoneme)
        # 用这个新的记忆，把所有的单词拼写一遍
        improvement = evaluate_improvement(forget_memory_df, t)
        improvement.generate_answer()
        improvement_acc, improvement_com, improvement_per = improvement.evaluation()
        print(f'{key}和{value}结果是{improvement_acc},{improvement_com},{improvement_per}')
        improvement_acc_list.append(improvement_acc)
        improvement_com_list.append(improvement_com)
        improvement_per_list.append(improvement_per)
        kl_divergence.append(value[0])
        accuracy_list.append(value[1])
    normal_kl_divergence = [round(i/(2*max(kl_divergence)), 3) for i in kl_divergence]
    print(accuracy_list)
    print(normal_kl_divergence)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(improvement_acc_list, label='improvement')
    axs[0].plot(normal_kl_divergence, label='KL_divergence')
    axs[0].plot(accuracy_list, label='avg_accuracy')
    axs[0].plot(excellent_acc_list * n, label='excellent_accuracy')
    axs[0].plot(forget_acc_list * n, label='forget_accuracy')
    axs[0].legend(loc='upper right')
    # axs[1].plot(improvement_com_list)
    # axs[1].plot(excellent_com_list * n)
    # axs[1].plot(forget_com_list * n)
    #
    # axs[2].plot(improvement_per_list)
    # axs[2].plot(excellent_per_list * n)
    # axs[2].plot(forget_per_list * n)

    plt.show()
