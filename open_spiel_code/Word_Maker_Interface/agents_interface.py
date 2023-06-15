"""
1: agent base abstract class, and per agent interface
2: initial some parameters that will be regularly used in agents instance
"""

import abc
from typing import List, Tuple, Dict, Union


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
                    player_name: string. Defaults to `student agent`.
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


class ChanceInterface(AgentAbstractBaseClass):
    """
    formulate the chance player interface
    step 1: Chance Player get the pairs of input and output [chinese, phonetic] -> [spelling]
    step 2: Chance Player randomly select a pair of [chinese, phonetic] -> [spelling]
    step 3: if the game terminates, it is either terminate or to step 2
    :return: the one pair of [chinese, phonetic] -> [spelling]
    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 tasks_pool: Dict[str, str]):
        super().__init__(player_id, player_name)
        """Initializes chance agent.

        Args:
            tasks_pool: Dictionary, mandatory {'人的 h j u m ʌ n': 'h u m a n', ...............}.
            self._ch_pho: string, '人的 h j u m ʌ n'
            self._word: string. 'h u m a n'

        """
        self._tasks_pool: Dict[str, str] = tasks_pool
        self._ch_pho: str = ''
        self._word: str = ''

    @abc.abstractmethod
    def select_word(self) -> Tuple[str, str]:
        """
            :return: the task pair
            """
        return self._ch_pho, self._word


class TutorInterface(AgentAbstractBaseClass):
    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str):
        super().__init__(player_id, player_name)
        """Initializes tutor agent.

        Args:
            self.difficulty_levels_definition: Dictionary, mandatory, the difficulty definition
            self.current_difficulty_level: integer, the initial difficulty level
            self.legal_difficulty_level: store available letter 

        """
        self.difficulty_levels_definition: Dict[int, Dict[str, int]] = {}
        self.current_difficulty_level: int = 1
        self.legal_difficulty_level: List[int] = []

    @abc.abstractmethod
    def legal_difficulty_levels(self,
                                previous_difficulty_level: int) -> List[int]:
        """
        :return: the accessible index of difficulty level, keep or upgrade difficulty level
        """
        pass

    @abc.abstractmethod
    def decide_difficulty_level(self,
                                current_game_round: int) -> Dict[str, int]:
        """
        :return: the difficulty setting
        """
        pass


class StudentInterface(AgentAbstractBaseClass):
    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 chinese_phonetic: str,
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
            self.masks, tensor, will be used in spelling 
            self.stu_spelling, string, initialize student spelling
        """
        self._CONFUSING_LETTER_DIC: Dict[str, List[str]] = {}
        self.chinese_phonetic: str = chinese_phonetic
        self.difficulty_setting: Dict[str, int] = difficulty_setting
        self.target_length = len(target_english.replace(" ", ""))
        self.available_letter: List[str] = target_english.split(' ')
        self.confusing_letter: List[str] = []
        self.masks = None
        self.stu_spelling: str = ''
        self.stu_feedback: Union[None, Dict[str, int]] = {}

    @abc.abstractmethod
    def letter_space(self) -> List[str]:
        """
            :return: list of legal letters [a,c,d,f,e,b,t,y]
        """
        pass

    @abc.abstractmethod
    def student_spelling(self,
                         student_feedback:  Union[None, Dict[str, int]]) -> List[str]:
        """
        :return: student spelling
        """
        pass


class ExaminerInterface(AgentAbstractBaseClass):
    """
    input: students spelling, correct spelling

        :return:  students feedback: {letter, color_index}
                  tutor feedback: [accuracy, completeness, attempts,....]
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
                  tutor feedback: [accuracy, completeness, attempts,....]
        """
        pass
