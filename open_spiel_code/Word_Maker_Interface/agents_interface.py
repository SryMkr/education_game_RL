"""
define every agent
"""

import abc
from typing import List, Tuple, Dict

# 对于我自己的情况，我得创建一个字典，每一个时间步长得agent是谁，agent得动作


class ChanceInterface(metaclass=abc.ABCMeta):
    """
    formulate the chance player interface
    step 1: Chance Player get the pairs of input and output [chinese, phonetic] -> [spelling]
    step 2: Chance Player randomly select a pair of [chinese, phonetic] -> [spelling]
    step 3: if the game terminates, it is either terminate or to step 2
    :return: the one pair of [chinese, phonetic] -> [spelling]
    """

    @abc.abstractmethod
    def __init__(self, tasks_pool: Dict[str, str], player_id: int, player_name: str = 'student agent'):
        """Initializes agent.

        Args:
            tasks_pool: Dictionary, mandatory {'人的 h j u m ʌ n': 'h u m a n', ...............}.
            player_id: integer, mandatory.
            player_name: string. Defaults to `student agent`.

        """
        self.tasks_pool: Dict[str, str] = tasks_pool  # {'人的 h j u m ʌ n': 'h u m a n', ...............}
        self.ch_pho: str = ''  # store the chinese and phonetic '人的 h j u m ʌ n'
        self.word: str = ''  # store the english spelling 'h u m a n'
        self._player_id: int = player_id  # agent ID
        self._player_name: str = player_name  # agent name

    @property
    def player_id(self) -> int:
        """
        :return: the player_id
        """
        return self._player_id

    @property
    def legal_tasks(self) -> Dict[str, str]:
        """
               :return: the legal tasks pool
               """
        return self.tasks_pool

    @property
    @abc.abstractmethod
    def select_word(self) -> Tuple[str, str]:
        """
        select the task
            :return: the task pair
            """
        return self.ch_pho, self.word


class TutorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        """
        define difficulty level
        """
        self.difficulty_levels_definition: Dict[int, Dict[str, int]] = {}  # define the difficulty level
        self.current_difficulty_level: int = 1  # initial difficulty level

    @abc.abstractmethod
    def legal_difficulty_levels(self, previous_difficulty_level: int) -> List[int]:
        """
        :return: the accessible index of difficulty level, keep or upgrade difficulty level
        """
        pass

    @abc.abstractmethod
    def decide_difficulty_level(self, current_game_round: int) -> Dict[str, int]:
        """
        :return: the difficulty setting
        """
        pass


class StudentInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, chinese_phonetic: str, target_english: str, difficulty_setting: Dict[str, int]):
        """
        student agent should know the chinese, phonetic, and the target length
        """
        self._CONFUSING_LETTER_DIC: Dict[str, List[str]] = {}
        self.chinese_phonetic: str = chinese_phonetic
        self.difficulty_setting: Dict[str, int] = difficulty_setting
        self.target_length = len(target_english.replace(" ", ""))  # get the length of target english
        self.available_letter: List[str] = target_english.split(' ')
        self.confusing_letter: List[str] = []
        self.masks = None

    @abc.abstractmethod
    def letter_space(self) -> List[str]:
        """
            # 不管迷惑字母也不管可选择的字母，现在需要学生先给一个答案出来
            合法的动作包括：正确的拼写字母，迷惑字母，以及选择了一个动作以后，要将该字母从库中删除
            get legal actions from chance player
            :return: list of legal letters [a,c,d,f,e,b,t,y]
        """
        pass

    @abc.abstractmethod
    def student_spelling(self, student_feedback: List[int]) -> List[str]:
        """
        :return: student spelling
        """
        pass


class ExaminerInterface(metaclass=abc.ABCMeta):
    """
    input: students spelling, correct spelling

        :return:  students feedback: {letter, color}
                  tutor feedback: [accuracy, completeness, attempts,....]
        """

    def __init__(self):
        """
            initialize the student feedback, tutor feedback
                """
        self.student_feedback: Dict[str, int] = {}
        self.tutor_feedback: List[float] = []

    @abc.abstractmethod
    def give_feedback(self, student_spelling: str, correct_spelling: str) -> Tuple[Dict[str, int], List[float]]:
        """
        :return:  students feedback: {letter, color}
                  tutor feedback: [accuracy, completeness, attempts,....]
        """
        pass
