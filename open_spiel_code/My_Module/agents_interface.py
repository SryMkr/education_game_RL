"""
define what can be observed and actions per agent
"""

import abc
from typing import List, Tuple, Dict
import enum


class ChanceInterface(metaclass=abc.ABCMeta):
    """
    formulate the chance player interface
    step 1: Chance Player get the pairs of input and output [chinese, phonetic] -> [spelling]
    step 2: Chance Player randomly select a pair of [chinese, phonetic] -> [spelling]
    step 3: if the game terminates, it is either terminate or to step 2
    :return: the one pair of [chinese, phonetic] -> [spelling]
    游戏轮数增加有两种情况：（1）拼对且小于等于总机会次数（2）最后一次机会也拼错
    """

    @abc.abstractmethod  # observe tasks
    def __init__(self, tasks_pool: Dict[str, str], maximum_game_rounds: int = 4):
        """
            initialize : input: tasks pool
            :return        one of the tasks
               """
        self.tasks_pool: Dict[str, str] = tasks_pool  # {'人的 h j u m ʌ n': 'h u m a n', ...............}
        self.ch_pho: str = ''  # store the chinese and phonetic
        self.word: str = ''  # store the english spelling
        self.maximum_game_rounds: int = maximum_game_rounds  # the total game rounds

    @property
    def legal_tasks(self) -> Dict[str, str]:
        """
               :return: the legal tasks pool
               """
        return self.tasks_pool

    @abc.abstractmethod
    def select_word(self) -> Tuple[str, str]:
        """
        select the task
            :return: the task pair
            """
        return self.ch_pho, self.word

    def is_terminal(self, current_game_round) -> bool:
        """
                if game terminated or not
                    :return: true/false
                    """
        if current_game_round == self.maximum_game_rounds:
            return True
        else:
            return False


class TutorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        """
        define difficulty level
        """
        self.difficulty_levels_definition: Dict[int, Dict[str, int]] = {}  # define the difficulty level
        self.difficulty_level: int = 1  # initial difficulty level

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

    # @abc.abstractmethod
    # def decide_action(self, previous_difficulty_level: int, completeness: float, accuracy: float) -> int:
    #     """
    #     :return: the index of difficulty level
    #     """
    #     pass
    # @property
    # def current_game_round(self) -> int:
    #     """
    #     :return: current round
    #     """
    #     pass
    #
    # @property
    # def is_round_finished(self) -> bool:
    #     """
    #     :return: is current round finished?
    #     """
    #     pass


# class StudentActionSpace(enum.IntEnum):
#     a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = \
#         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25


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

    @abc.abstractmethod
    def letter_space(self) -> List[str]:
        """
            # 不管迷惑字母也不管可选择的字母，现在需要学生先给一个答案出来
            合法的动作包括：正确的拼写字母，迷惑字母，以及选择了一个动作以后，要将该字母从库中删除
            get legal actions from chance player
            :return: list of legal actions [a,c,d,f,e,b,t,y]
        """
        pass

    @abc.abstractmethod
    def student_spelling(self) -> List[str]:
        """
        :return: student spelling
        """
        pass

    # @property
    # def get_word_length(self) -> int:
    #     """
    #         get word length from chance player
    #         :return: word length
    #     """
    #     pass
    #
    # @property
    # def get_total_attempts(self) -> int:
    #     """
    #         get total attempts from tutor player
    #         :return: total attempts
    #     """
    #     pass
    #
    # @abc.abstractmethod
    # def get_feedback(self) -> List[int]:
    #     """
    #         :return: get feedback from examiner  feedback [GREEN, YELLOW, RED]
    #     """
    #     pass
    #

    #
    # @abc.abstractmethod
    # def is_attempts_finished(self) -> bool:
    #     """
    #     = word length
    #     students spell finished or not
    #     :return: true/false
    #     """
    #     pass


class ExaminerInterface(metaclass=abc.ABCMeta):
    """
    1: examiner should get the correct spelling, and student spelling
    2: examiner should mark students spelling
    :return feedback： accuracy, completeness, red, green, yellow
    """

    @abc.abstractmethod
    def give_feedback(self, student_spelling: str, correct_spelling: str):
        """
        :return: feedback [GREEN, YELLOW, RED]
        """
        pass
