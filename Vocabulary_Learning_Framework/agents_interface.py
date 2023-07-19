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
                    :return: one session words data over time
                    """
        pass


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
    """
    input: students spelling, correct spelling

        :return:  students feedback: {letter, color_index}
                  tutor feedback: [accuracy, completeness, attempts, difficulty_level....]
        """

    def __init__(self,
                 player_id: int,
                 player_name: str):
        super().__init__(player_id, player_name)
        """Initializes examiner agent.

        Args:
            self.student_feedback: Dictionary, mandatory, store students' feedback
            self.tutor_feedback: list, store tutor feedback
        """
        self.student_feedback: Dict[str, int] = {}
        self.tutor_feedback: List[float] = []

    @abc.abstractmethod
    def give_feedback(self,
                      student_spelling: str,
                      correct_spelling: str) -> Tuple[Dict[str, int], List[float]]:
        """
        :return:  students feedback: {letter, color_index}
                  tutor feedback: [accuracy, completeness, attempts, difficulty_level....]
        """
        pass
