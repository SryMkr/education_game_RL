"""
1: agents instance
2: the method of tutor should be changed in the future
"""

from agents_interface import *
import random


# TaskCollector Agent
class SessionCollectorPlayer(SessionCollectorInterface):
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
        action = 0
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        if self._policy == 'random':
            action = random.choice(legal_actions)  # random
        elif self._policy == 'sequential':
            action = legal_actions[0]  # sequential
        return action


# PresentWord Agent
class PresentWordPlayer(PresentWordInterface):
    def __init__(self,
                 player_id,
                 player_name,
                 policy):
        super().__init__(player_id,
                         player_name,
                         policy)

    def action_policy(self, time_step):
        action = 0
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        # 用四种不同的方式选择单词的长度，选择了以后要删除
        if self._policy == 'random':
            action = random.choice(legal_actions)  # 反正都是都是随机，根本无所谓
        if self._policy == 'sequential':
            action = legal_actions[0]  # 直接选择第一个就可以
        if self._policy == 'easy_to_hard':
            action = sorted(legal_actions)[0]  # 按照长度从简单到难做一个排序，然后选择第一个
        if self._policy == 'DDA':  # RL!!!!!!!!! need to be finished
            pass
        return action

    def step(self, time_step):
        action = self.action_policy(time_step)
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

    def stu_spell(self, time_step) -> List[int]:
        """
        :return: student spelling
        """
        actions = []
        condition = time_step.observations["condition"]
        answer_length = time_step.observations["answer_length"]
        legal_actions = time_step.observations["legal_actions"][self.player_id]

        if self._policy == 'random':
            for letter_index in range(answer_length):
                selected_index = random.choice(legal_actions)
                actions.append(selected_index)

        elif self._policy == 'perfect':
            pass
        elif self._policy == 'forget':
            pass
        return actions

    def stu_learn(self, time_step) -> None:
        """
        update n-grams based on feedback!!!!!!!!!!!!!!!!
        """

    def step(self, time_step):
        self.stu_learn(time_step)
        actions = self.stu_spell(time_step)
        return actions

class ExaminerPlayer(ExaminerInterface):
    def __init__(self,
                 player_id,
                 player_name):
        super().__init__(player_id,
                         player_name)

    def step(self, time_step):
        actions = []
        answer = time_step.observations["answer"].split(' ')  # 'b a t h' --> ['b', 'a', 't', 'h']
        student_spelling = time_step.observations["student_spelling"]  # ['f', 'h', 'v', 'q']
        answer_length = time_step.observations["answer_length"]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        for position in range(answer_length):
            if student_spelling[position] == answer[position]:
                actions.append(legal_actions[1])
            else:
                actions.append(legal_actions[0])
        return actions

