"""
1: agent base abstract class, and per agent interface
2: initial some parameters that will be regularly used in agents instance
# In a nutshell: An agent normally has
    (1) attributes: player_ID, player_Name, **agent_specific_kwargs, all implemented in __init__ function
    (2) step function: A: parameter: get the observation of environment time step
                       B: a policy get the observation and provide the action probabilities, then agent select an action based on probabilities
                       (observation->action probabilities->action)
    compared with the (uniform_random) agent that have different policy


思考将agent，可以分为直接初始化，然后结合策略，选择动作的道路
"""

import abc
from typing import List, Tuple, Dict
import random


class AgentAbstractBaseClass(metaclass=abc.ABCMeta):
    """Abstract base class for all agents."""

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 **agent_specific_kwargs):
        """Initializes agent.

                Args:
                    player_id: zero-based integer，for index agent
                    player_name: string.
                    **agent_specific_kwargs: optional extra args.
                """
        self._player_id: int = player_id
        self._player_name: str = player_name

    @abc.abstractmethod
    def step(self, time_step):
        """
           Agents should handle `time_step` and extract the required part of the
           `time_step.observations` field.

           Arguments:
             time_step: an instance of rl_environment.TimeStep.
           Returns:
             A `StepOutput` for the current `time_step`. (an action or information) !!!!!!!!!!!
           """

    @property
    def player_id(self) -> int:
        """
        :return: the player_id
        """
        return self._player_id

    @property
    def player_name(self) -> str:
        """
        :return: the player_name
        """
        return self._player_name


class SessionCollectorInterface(AgentAbstractBaseClass):
    """
    是为了组合学习新单词和组合旧单词的功能
    怎么能够增加一些随机性呢？让每个agent都有一定的策略可以使用，确定展示那个session呗,如何构造legal actions?
    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str
                 ):
        super().__init__(player_id, player_name)
        """Initializes TaskCollector agent.
        :arg
            self._action: the selected action
        """
        self._action: int = 0

    @abc.abstractmethod
    def step(self, time_step) -> List[List[str]]:
        """
                    :return: the number of selected session
                    """
        pass


class PresentWordInterface(AgentAbstractBaseClass):
    """ select one word from session data, there are four method (1) sequential (2) random (3) easy to hard (4)DDA"""

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 selection_method: str,
                 ):
        super().__init__(player_id, player_name)
        '''
        :args:
               
                self._task: a specific task for student agent
                self._selection_method: the method of selecting task 
                self._difficulty_session_data: sort from easy to hard
                self._shuffle_list: copy original data to shuffle
        '''

        self._task: List[str] = []
        self._selection_method = selection_method
        self._difficulty_session_data = sorted(self._session_data, key=lambda x: len(x[-1]))
        self._shuffle_list = self._session_data[:]
        random.shuffle(self._shuffle_list)

    def sequential_method(self, session_data):
        """ sequence method
          对于顺序选择来说，将第一个选择出来并移到末尾，如果正确了将其移除就好，反正永远只选第一个"""
        self._task = session_data[0]
        self._session_data = session_data[1:] + [session_data[0]]

    def random_method(self, session_data):
        """ random method"""
        self._task = self._shuffle_list[0]
        self._shuffle_list = self._shuffle_list[1:] + [self._shuffle_list[0]]
        self._session_data = self._shuffle_list

    def easy_to_hard(self, session_data):
        """ the definition of difficulty of task is the length of task
           对于从简单到难来说，先按照字母长度排序，然后就是按照顺序选择，直到答对"""
        self._task = self._difficulty_session_data[0]
        self._difficulty_session_data = self._difficulty_session_data[1:] + [self._difficulty_session_data[0]]
        self._session_data = self._difficulty_session_data

    def DDA(self, session_data):
        """ the dynamic difficulty adjustment method, do not achieve currently"""
        pass

    @property
    def session_data(self) -> List[List[str]]:
        return self._session_data

    def step(self, time_step) -> (List[List[str]], List[str]):
        """
        if correctly answer at hardest test level, the agent need to remove the task from session
        :rtype: object
        :returns the task"""

        if self._selection_method == 'sequential':
            self.sequential_method(time_step.observations["vocab_session"])
        elif self._selection_method == 'random':
            self.random_method(time_step.observations["vocab_session"])
        elif self._selection_method == 'easy_to_hard':
            self.easy_to_hard(time_step.observations["vocab_session"])
        elif self._selection_method == 'DDA':
            self.DDA(time_step.observations["vocab_session"])
        return self._session_data, self._task


'''
class TutorInterface(AgentAbstractBaseClass):
    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 tasks_pair: List[Tuple[str, str]]):
        super().__init__(player_id, player_name)
        """Initializes tutor agent.

        Args:
            self.difficulty_levels_definition: Dictionary, mandatory, the difficulty definition
            self.current_difficulty_level: List[integer], the initial difficulty level is always 1 
            self.legal_difficulty_level: store legal difficulty level 
        1： 老师是肯定提前分配好所有的难度的，也就是每个任务都有一个难度
        2： 那我的每个任务都需要有一个合法的难度范围
        """
        self.difficulty_levels_definition: Dict[int, Dict[str, int]] = {}
        self.current_difficulty_level: Dict[str, int] = {}
        self.legal_difficulty_level: Dict[str, List[int]] = {}

    @abc.abstractmethod
    def legal_difficulty_levels(self) -> Dict[str, List[int]]:
        """
        :return: the accessible index of difficulty level, keep or upgrade difficulty level
        """
        pass

    @abc.abstractmethod
    def decide_difficulty_level(self,
                                **agent_specific_kwargs,
                                ) -> Dict[str, int]:
        """
        #  [self, state]-> action -> difficulty setting
        # get the state of environment (parameter), implemented policy, select an action from legal action,
        then output the selected action(difficulty level)
        :return: the difficulty setting
        """
        pass
'''


class StudentInterface(AgentAbstractBaseClass):
    """ student agent can see conditions(chinese,phonetic,POS) -> provide spelling,
    optional information: answer length, available letter,
    ['蜘蛛 n s p aɪ d ɝ', 's p i d e r']
    至少可以先实现一个随机的拼写，然后把与环境的交互定义好，完成一次整体的流程
    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 stu_type: str,
                 word_length: int,
                 spelling_condition: str,
                 stu_feedback: List[str]):
        super().__init__(player_id, player_name)
        """Initializes student agent.

        Args:
            self._spelling_condition: str, mandatory, what condition (chinese,phonetic,POS) does student agent can see
            self._stu_spelling: student provide spelling based on condition
            self._stu_learn: student train n-grams based on feedback
            self._letter_space: available letter
            self._stu_type: student type: random, perfect, forget
            self._word_length: int, word length for control random student
        """
        self._spelling_condition: str = spelling_condition
        self._stu_spelling: List[str] = []
        self._stu_feedback: List[str] = stu_feedback
        self._letter_space: List[str] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                         'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self._stu_type = stu_type
        self._word_length = word_length

    def random_stu(self):
        """student does learn and forget"""
        for letter_position in range(self._word_length):
            letter = random.choice(self._letter_space)
            self._stu_spelling.append(letter)

    def perfect_stu(self):
        """student learn but does not forget"""
        pass

    def forget_stu(self):
        """student learn but forget: (1) be a perfect student (2) add noise to simulate forget overtime"""
        pass

    @abc.abstractmethod
    def stu_spelling(self) -> List[str]:
        """
        :return: student spelling
        """
        if self._stu_type == 'random':
            self.random_stu()
        elif self._stu_type == 'perfect':
            self.perfect_stu()
        elif self._stu_type == 'forget':
            self.forget_stu()
        return self._stu_spelling

    @abc.abstractmethod
    def stu_learn(self) -> None:
        """
        update n-grams based on feedback
        """
        pass


class ExaminerInterface(AgentAbstractBaseClass):
    """Examiner feedback"""

    def __init__(self,
                 player_id: int,
                 player_name: str):
        super().__init__(player_id, player_name)
        """Initializes examiner agent.
        Args:
            self.examiner_feedback: [accuracy, completeness, letters]
            
        """
        self.examiner_feedback: List[int, int, List[int]] = []

    @abc.abstractmethod
    def give_feedback(self,
                      student_spelling: str,
                      correct_spelling: str) -> Tuple[Dict[str, int], List[float]]:
        """
          :return:  Examiner feedback: [accuracy, completeness, letters]
        """
        pass
