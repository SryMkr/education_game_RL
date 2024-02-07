"""
1: four agents instance

Tasks:
(1) simulate forgetting students
(2) student learn from feedback
(3) optimise all agents
"""

import os
import random
import string
import Levenshtein
from agents_interface import *
import pandas as pd
import torch

# TaskCollector Agent
class SessionCollectorPlayer(SessionCollectorInterface):
    """对于挑选session来说，没有可进化的空间，要么直接session也不挑了，在词库中进行选择"""

    def __init__(self,
                 player_id,
                 player_name,
                 policy):
        super().__init__(player_id,
                         player_name,
                         policy)

    def step(self, time_step) -> int:
        """
                    :return: the action index
                    """
        action = 0  # 默认action一开始就是0
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        if self._policy == 'random':  # 随机挑选session
            action = random.choice(legal_actions)  # random
        elif self._policy == 'sequential':  # 按照顺序挑选session，每次都选择首位,因为选过的已经删除
            action = legal_actions[0]  # sequential
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
        STU_MEMORY_PATH = os.path.join(CURRENT_PATH, 'forgetting_memory.xlsx')
        self.stu_memory_df = pd.read_excel(STU_MEMORY_PATH, index_col=0, header=0)  # excellent students
        self.stu_memory_tensor = torch.tensor(self.stu_memory_df.values, dtype=torch.float32)  # the shape of distribution

    def stu_spell(self, time_step) -> List[int]:
        """  三类学生，随机，优秀，遗忘
        :return: student spelling
        """
        actions = []
        condition = time_step.observations["condition"].split(' ')  # 只有音标
        answer_length = time_step.observations["answer_length"]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        alphabet = string.ascii_lowercase

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
                    possible_results = self.stu_memory_df.loc[self.position_condition[0], result_columns]
                    letter = possible_results.idxmax()
                else:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_memory_df.loc[self.position_condition, result_columns]
                    letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                    letter = letters_prob.idxmax()
                spelling.append(letter)

            for letter_position in spelling:
                actions.append(alphabet.index(letter_position.split('_')[0]))

        elif self._policy == 'forget':
            """wait to be achieved,把遗忘的算法今天嵌入进去，不难，直接改过来就行 """
            noise = torch.randn_like(self.stu_memory_tensor)
            new_memory_tensor = 0.9998 * self.stu_memory_tensor + 0.0218 * noise  # how to forget
            result_df = pd.DataFrame(new_memory_tensor.numpy(), index=self.stu_memory_df.index,
                                     columns=self.stu_memory_df.columns)
            self.stu_memory_df = result_df.mask(result_df <= 0, 0.0001)
            spelling = []  # store the letter_position
            self.position_condition = []  # empty is every time
            for position, phoneme in enumerate(condition):
                self.position_condition.append(phoneme + '_' + str(position))
            for i in range(answer_length):
                if i == 0:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_memory_df.loc[self.position_condition[0], result_columns]
                    letter = possible_results.idxmax()
                else:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_memory_df.loc[self.position_condition, result_columns]
                    letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                    letter = letters_prob.idxmax()
                spelling.append(letter)

            for letter_position in spelling:
                actions.append(alphabet.index(letter_position.split('_')[0]))
        return actions

    def stu_learn(self, time_step) -> None:
        """
        update n-grams based on feedback!!!!!!!!!!!!!!!! no need to learn currently
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
        actions = []
        answer = ''.join(time_step.observations["answer"].split(' '))  # 'b a t h' --> ['b', 'a', 't', 'h']
        student_spelling = ''.join(time_step.observations["student_spelling"])  # ['f', 'h', 'v', 'q']
        word_accuracy = round(Levenshtein.ratio(answer, student_spelling), 3)
        word_completeness = round(1 - Levenshtein.distance(answer, student_spelling) / len(answer), 3)
        answer_length = time_step.observations["answer_length"]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        for position in range(answer_length):
            if student_spelling[position] == answer[position]:
                actions.append(legal_actions[1])
            else:
                actions.append(legal_actions[0])
        self.accuracy.append(word_accuracy)
        return student_spelling, actions, word_accuracy, word_completeness
