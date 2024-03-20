"""
1: four agents instance

Tasks:
(1) simulate forgetting students
(2) student learn from feedback
(3) optimise all agents
2/21 implement the multi armed bandit into session collector player
step 1: calculate the KL divergence between wrong letter and excellent
step 2: sort by descending
step 3：the order will be the top 50 words
既然分组了，如何添加噪声，很简单，按照session的个数，把噪声分组，那么对应每一组搁的时间代表了那一组的遗忘，不是全部添加而是只添加那几个单词
新的单词，按照现在有方式进行软更新，看看最后的结果能不能抵抗遗忘？
至于老师推荐单词的话，那是后期的任务了，主要是为了一个平衡，对于每一个推荐的单词，准确度都要是当前组里最高的
"""
from itertools import chain

import os
import random
import string
import Levenshtein
from agents_interface import *
import pandas as pd
import torch
import numpy as np


# TaskCollector Agent
class SessionCollectorPlayer(SessionCollectorInterface):
    """select the prioritised words"""
    def __init__(self,
                 player_id,
                 player_name,
                 policy):
        super().__init__(player_id,
                         player_name,
                         policy)

    def step(self, time_step):
        """
                    :return: the words need to be reviewed
                    """
        action = []
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        review_words_number = time_step.observations["review_words_number"]
        history_information = time_step.observations["history_information"]
        if self._policy == 'random':  # randomly select one session
            action = random.sample(legal_actions, review_words_number)
        elif self._policy == 'MAB':  # Multi-Arm Bandit algorithm
            # print(legal_actions)
            # history information will be used here
            # apparently the history information is the agent can observe
            action = legal_actions[0]  # 给列表排序，并且选择前50个
        return action


# PresentWord Agent
class PresentWordPlayer(PresentWordInterface):
    """"""

    def __init__(self,
                 player_id,
                 player_name,
                 policy):
        super().__init__(player_id,
                         player_name,
                         policy)

    def define_difficulty(self, time_step) -> Dict[tuple, int]:
        """tutor agent define the difficulty of each task
        difficulty definition: the length of word
        """
        # 循环所有的任务，并且计算每个任务的长度，
        task_difficulty = {}
        # 合法的字符是个列表[音标，字母]
        legal_actions: List[str] = time_step.observations["legal_actions"][self.player_id]
        for task in legal_actions:
            task_difficulty[tuple(task)] = len(''.join(task[1].split(' ')))
        return task_difficulty

    def action_policy(self, task_difficulty):
        action = ''  # action is empty string
        # 用四种不同的方式选择单词的长度，选择了以后要删除
        if self._policy == 'random':
            action = random.choice(list(task_difficulty.keys()))
        if self._policy == 'sequential':
            action = next(iter(task_difficulty.items()))[0]
        if self._policy == 'easy_to_hard':
            action = min(task_difficulty.items(), key=lambda x: x[1])[0]
        if self._policy == 'DDA':  # RL!!!!!!!!! need to be finished，直接搞网络，策略网络来做
            pass
        return list(action)

    def step(self, time_step):
        task_difficulty = self.define_difficulty(time_step)
        action = self.action_policy(task_difficulty)
        return action


# student player
class StudentPlayer(StudentInterface):
    def __init__(self,
                 player_id,
                 player_name,
                 policy):
        super().__init__(player_id,
                         player_name,
                         policy)

        CURRENT_PATH = os.getcwd()  # get the current path
        self.policy = policy
        STU_MEMORY_PATH = os.path.join(CURRENT_PATH, 'agent_RL/excellent_memory.xlsx')
        self.stu_forget_df = pd.read_excel(STU_MEMORY_PATH, index_col=0, header=0)  # for forgetting
        self.stu_excellent_df = pd.read_excel(STU_MEMORY_PATH, index_col=0, header=0)  # excellent students memory
        self.stu_memory_tensor = torch.tensor(self.stu_excellent_df.values,
                                              dtype=torch.float32)  # the shape of distribution
        self.noise = torch.randn_like(self.stu_memory_tensor)  # generate the noise
        self.scaled_noise = (self.noise - self.noise.min()) / (self.noise.max() - self.noise.min())
        self.scaled_noise_df = pd.DataFrame(self.scaled_noise.numpy(), index=self.stu_excellent_df.index,
                                            columns=self.stu_excellent_df.columns)
        self.current_session_num = 0  # 初始化

    @staticmethod
    def forgetting_parameters(timing_steps, excel_start=0.9, excel_end=0.4, noise_start=0.1, noise_end=1, decay_rate=2):
        """the function aims to simulate the forgetting curve"""
        timing_points = np.linspace(0, 1, timing_steps)
        excel_list = (excel_start - excel_end) * np.exp(-decay_rate * timing_points) + excel_end
        noise_list = (noise_end - noise_start) * (1 - np.exp(-decay_rate * timing_points)) + noise_start
        return excel_list, noise_list

    @staticmethod
    def forget_process(unique_phonemes, excellent_dataframe, noise_dataframe, excel_ratio, random_ratio):
        """直接改变一个session的记忆"""
        excellent_dataframe_copy = excellent_dataframe.copy()
        for pho in unique_phonemes:
            excellent_dataframe_copy.loc[pho] = excel_ratio * excellent_dataframe.loc[pho] + random_ratio * \
                                                noise_dataframe.loc[pho]
        result_df = excellent_dataframe_copy.div(excellent_dataframe_copy.sum(axis=1), axis=0)  # normalize
        return result_df

    def stu_spell(self, time_step) -> List[int]:
        """  三类学生，随机，优秀，遗忘
        :return: student spelling
        """
        actions = []
        sessions_num = time_step.observations["sessions_number"]
        excel_list, noise_list = self.forgetting_parameters(sessions_num)
        condition = time_step.observations["condition"].split(' ')  # 只有音标
        answer_length = time_step.observations["answer_length"]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        alphabet = string.ascii_lowercase

        if (time_step.observations["current_session_num"] - self.current_session_num == 2) and self.policy == "forget":
            # 执行遗忘操作，读取历史信息，并且按照历史信息中的音标来遗忘
            history_information = time_step.observations["history_information"]
            # 要给每一个phoneme加上标签再分割
            split_phonemes = [key[0].split(' ') for key in history_information.keys()]
            position_phonemes = []
            for phonemes_list in split_phonemes:
                for index, value in enumerate(phonemes_list):
                    position_phonemes.append(value + '_' + str(index))

            unique_phonemes = set(position_phonemes)

            self.stu_forget_df = self.forget_process(list(unique_phonemes), self.stu_excellent_df, self.scaled_noise_df,
                                                     excel_list[self.current_session_num],
                                                     noise_list[self.current_session_num])
            # print(self.stu_forget_df.shape)
        self.current_session_num = time_step.observations["current_session_num"] - 1

        if self._policy == 'random':
            for letter_index in range(answer_length):
                selected_index = random.choice(legal_actions)
                actions.append(selected_index)

        elif self._policy == 'excellent':
            """use maximum expectation algorithm"""
            spelling = []  # store the letter_position
            self.position_condition = []  # empty is every time
            for position, phoneme in enumerate(condition):
                self.position_condition.append(phoneme + '_' + str(position))
            for i in range(answer_length):
                if i == 0:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_excellent_df.loc[self.position_condition[0], result_columns]
                    letter = possible_results.idxmax()
                else:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_excellent_df.loc[self.position_condition, result_columns]
                    letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                    letter = letters_prob.idxmax()
                spelling.append(letter)

            for letter_position in spelling:
                actions.append(alphabet.index(letter_position.split('_')[0]))

        elif self._policy == 'forget':

            """每一轮，按照历史记录添加噪声，代表每一轮之后都会进行遗忘操作"""

            spelling = []  # store the letter_position
            self.position_condition = []  # empty is every time
            for position, phoneme in enumerate(condition):
                self.position_condition.append(phoneme + '_' + str(position))

            for i in range(answer_length):
                if i == 0:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_forget_df.loc[self.position_condition[0], result_columns]
                    letter = possible_results.idxmax()
                else:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_forget_df.loc[self.position_condition, result_columns]
                    letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                    letter = letters_prob.idxmax()
                spelling.append(letter)

            for letter_position in spelling:
                actions.append(alphabet.index(letter_position.split('_')[0]))
        return actions

    def stu_learn(self, time_step) -> None:
        """
        update the forgetting matrix by soft updates
        """

    def step(self, time_step):
        # self.stu_learn(time_step)
        actions = self.stu_spell(time_step)
        return actions


class ExaminerPlayer(ExaminerInterface):
    def __init__(self,
                 player_id,
                 player_name):
        super().__init__(player_id,
                         player_name)
        self.accuracy = []

    def step(self, time_step):
        marks = []
        actions = {}
        answer = ''.join(time_step.observations["answer"].split(' '))  # 'b a t h' --> ['b', 'a', 't', 'h']
        student_spelling = ''.join(time_step.observations["student_spelling"])  # ['f', 'h', 'v', 'q']
        word_accuracy = round(Levenshtein.ratio(answer, student_spelling), 3)
        # not use in the future
        word_completeness = round(1 - Levenshtein.distance(answer, student_spelling) / len(answer), 3)
        answer_length = time_step.observations["answer_length"]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        for position in range(answer_length):
            if student_spelling[position] == answer[position]:
                marks.append(legal_actions[1])
            else:
                marks.append(legal_actions[0])
        self.accuracy.append(word_accuracy)
        # 在这里把位置加进去，然后和对错结合起来并组成一个元组
        for position, letter in enumerate(student_spelling):
            actions[letter + '_' + str(position)] = marks[position]
        return actions, word_accuracy, word_completeness
