"""
1: agent base abstract class, and per agent interface
2: initial some parameters that will be regularly used in agents instance
# In a nutshell: An agent normally has
    (1) attributes: player_ID, player_Name, **agent_specific_kwargs, all implemented in __init__ function
    (2) step function: A: parameter: get the observation of environment time step
                       B: a policy get the observation and provide the action probabilities, then agent select an action based on probabilities
                       (observation->action probabilities->action)
    compared with the (uniform_random) agent that have different policy
"""

import abc
from typing import List, Tuple, Dict, Optional
from torch import Tensor
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
                    player_id: integer, mandatory.
                    player_name: string.
                    **agent_specific_kwargs: optional extra args.
                """
        self._player_id: int = player_id
        self._player_name: str = player_name

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
    use 'random' or 'sequential' slice vocabulary data into sessions
    :return: self._vocabulary_session: sliced data by specific method
    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 vocabulary_data: List[List[str]],
                 new_words_number: int,
                 # review_words_number: int,
                 # review_selection_method: str
                 new_selection_method: str):
        super().__init__(player_id, player_name)
        """Initializes TaskCollector agent.

        Args:
            self._vocabulary_data: the whole vocabulary data
            self._new_words_number: the new words number in each session
            self._new_selection_method: the selection method regarding new words
            self._review_words_number: the review words number in each session
            self._review_selection_method: the selection method of review words
            self._vocabulary_session: sliced data by specific method
            self._current_session: after finishing each session, go next session
        """
        self._vocabulary_data: List[List[str]] = vocabulary_data
        self._new_words_number: int = new_words_number
        # self._review_words_number: int = review_words_number
        self._new_selection_method: str = new_selection_method
        # self._review_selection_method: str = review_selection_method
        self._vocabulary_session: List[List[List[str]]] = []

    def sequential_method(self):
        """the sequential method of new words"""
        for i in range(0, len(self._vocabulary_data), self._new_words_number):
            vocabulary_pieces = self._vocabulary_data[i:i + self._new_words_number]
            self._vocabulary_session.append(vocabulary_pieces)

    @abc.abstractmethod
    def random_method(self):
        """the random method of new words"""

    @abc.abstractmethod
    def piece_data(self):
        """
            当前先不考虑复习单词的事，就是单词表的挑选方式
            :return: the sessions with different methods
            """
        pass

    @abc.abstractmethod
    def session_collector(self, current_session) -> List[List[str]]:
        """
                    :return: select one session words data from vocabulary books over time
                    """
        pass


class PresentWordInterface(AgentAbstractBaseClass):
    """ select one word from session data, there are four method (1) sequential (2) random (3) easy to hard (4)DDA"""

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 session_data: List[List[str]],
                 selection_method: str,
                 ):
        super().__init__(player_id, player_name)
        '''
        :args:
                self._session_data: get from sessioncollector agent
                self._task: a specific task for student agent
                self._selection_method: the method of selecting task 
                self._difficulty_session_data: sort from easy to hard
                self._shuffle_list: copy original data to shuffle
        '''
        self._session_data = session_data
        self._task: List[str] = []
        self._selection_method = selection_method
        self._difficulty_session_data = sorted(self._session_data, key=lambda x: len(x[-1]))
        self._shuffle_list = self._session_data[:]
        random.shuffle(self._shuffle_list)

    def sequential_method(self):
        """ sequence method
          对于顺序选择来说，将第一个选择出来并移到末尾，如果正确了将其移除就好，反正永远只选第一个"""
        self._task = self._session_data[0]
        self._session_data = self._session_data[1:] + [self._session_data[0]]

    def random_method(self):
        """ random method"""
        self._task = self._shuffle_list[0]
        self._shuffle_list = self._shuffle_list[1:] + [self._shuffle_list[0]]
        self._session_data = self._shuffle_list

    def easy_to_hard(self):
        """ the definition of difficulty of task is the length of task
           对于从简单到难来说，先按照字母长度排序，然后就是按照顺序选择，直到答对"""
        self._task = self._difficulty_session_data[0]
        self._difficulty_session_data = self._difficulty_session_data[1:] + [self._difficulty_session_data[0]]
        self._session_data = self._difficulty_session_data

    def DDA(self):
        """ the dynamic difficulty adjustment method, do not achieve currently"""
        pass

    @property
    def session_data(self) -> List[List[str]]:
        return self._session_data

    def select_task(self) -> List[str]:
        """
        if correctly answer at hardest test level, the agent need to remove the task from session
        :returns the task"""

        if self._selection_method == 'sequential':
            self.sequential_method()
        elif self._selection_method == 'random':
            self.random_method()
        elif self._selection_method == 'easy_to_hard':
            self.easy_to_hard()
        elif self._selection_method == 'DDA':
            self.DDA()
        return self._task


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


class StudentInterface(AgentAbstractBaseClass):
    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 chinese_phonetic: str,  # ！！！！！！！直接掩盖掉，还是加一些噪声？
                 target_english: str,
                 difficulty_setting: Dict[str, int]):
        super().__init__(player_id, player_name)
        """Initializes student agent.

        Args:
            self._CONFUSING_LETTER_DIC: Dictionary, mandatory, the confusing letter definition
            self.chinese_phonetic: str, student can see the chinese and phonetic
            self.difficulty_setting: get the difficulty setting from tutor 
            self.target_length: integer, store the length of spelling
            self.available_letter: legal letters
            self.confusing_letter, the chosen confusing letter
            self.masks, tensor, will be used in spelling, learn from feedback 
            self.stu_spelling, string, initialize student spelling
            self.stu_feedback, Dict, getting from examiner
        """
        self._CONFUSING_LETTER_DIC: Dict[str, List[str]] = {}
        self.chinese_phonetic: str = chinese_phonetic
        self.difficulty_setting: Dict[str, int] = difficulty_setting
        self.target_length: int = len(target_english.replace(" ", ""))
        self.target_spelling: str = target_english
        self.available_letter: List[str] = target_english.split(' ')
        self.confusing_letter: List[str] = []
        self.masks: Optional[Tensor] = None
        self.stu_spelling: str = ''
        self.stu_feedback: Optional[Dict[str, int]] = {}

    @abc.abstractmethod
    def letter_space(self) -> List[str]:
        """
         only correct letter ot a group of confusing letter and correct letter
            :return: list of legal letters [a,c,d,f,e,b,t,y]
        """
        pass

    @abc.abstractmethod
    def student_spelling(self,
                         student_feedback: Optional[Dict[str, int]]) -> List[str]:
        """
        :return: student spelling
        """
        pass

    @abc.abstractmethod
    def student_memorizing(self) -> None:
        """
        :return: for students memorizing once difficulty level changes
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
            self.student_feedback: Dictionary, mandatory, store students' feedback
            self.tutor_feedback: list, store tutor feedback
        """
        self.examiner_feedback: List[int, int, List[int]] = []

    @abc.abstractmethod
    def give_feedback(self,
                      student_spelling: str,
                      correct_spelling: str) -> Tuple[Dict[str, int], List[float]]:
        """
          :return:  Examiner feedback: [accuracy, completeness, letter_judgement]
        """
        pass
