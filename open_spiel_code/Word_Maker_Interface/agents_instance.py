"""
1: agents instance
2: the method of tutor should be changed in the future
"""

from torch.utils.data import DataLoader
import random
from Word_Maker_RL.agents_interface import *
from student_spelling import evaluate, model, data_process, generate_batch
import Levenshtein as Levenshtein


# chance player
class ChancePlayer(ChanceInterface):
    def __init__(self, player_id, player_name, custom_tasks_pool):
        super().__init__(player_id, player_name, custom_tasks_pool)

    @property
    def select_word(self):
        self._ch_pho, self._word = random.choice(list(self._tasks_pool.items()))
        return self._ch_pho, self._word


# tutor player
class TutorPlayer(TutorInterface):
    def __init__(self, player_id, player_name):
        super().__init__(player_id, player_name)
        self.difficulty_level_definition = {
            1: {'attempts': 4, 'confusing_letter_setting': 0, 'chinese_setting': 1, 'phonetic_setting': 1},
            2: {'attempts': 3, 'confusing_letter_setting': 1, 'chinese_setting': 1, 'phonetic_setting': 1},
            3: {'attempts': 2, 'confusing_letter_setting': 1, 'chinese_setting': 1, 'phonetic_setting': 0},
            4: {'attempts': 1, 'confusing_letter_setting': 1, 'chinese_setting': 0, 'phonetic_setting': 1}}

    # get the list of difficulty level, the difficulty level should keep or upgrade
    def legal_difficulty_levels(self, previous_difficulty_level):
        difficulty_levels = [index for index, value in self.difficulty_level_definition.items()]
        self.legal_difficulty_level = difficulty_levels[(previous_difficulty_level - 1):]
        return self.legal_difficulty_level

    # get the difficulty setting
    def decide_difficulty_level(self, current_game_round):
        return self.difficulty_level_definition[current_game_round]


# student player
class StudentPlayer(StudentInterface):
    def __init__(self, player_id, player_name, chinese_phonetic, target_english, current_difficulty_setting):
        super().__init__(player_id, player_name, chinese_phonetic, target_english, current_difficulty_setting)

        self._CONFUSING_LETTER_DIC = {'a': ['e', 'i', 'o', 'u', 'y'], 'b': ['d', 'p', 'q', 't'],
                                      'c': ['k', 's', 't', 'z'], 'd': ['b', 'p', 'q', 't'],
                                      'e': ['a', 'o', 'i', 'u', 'y'], 'f': ['v', 'w'],
                                      'g': ['h', 'j'],
                                      'h': ['m', 'n'], 'i': ['a', 'e', 'o', 'y'], 'j': ['g', 'i'],
                                      'k': ['c', 'g'], 'l': ['i', 'r'], 'm': ['h', 'n'],
                                      'n': ['h', 'm'],
                                      'o': ['a', 'e', 'i', 'u', 'y'], 'p': ['b', 'd', 'q', 't'],
                                      'q': ['b', 'd', 'p', 't'], 'r': ['l', 'v'], 's': ['c', 'z'],
                                      't': ['c', 'd'], 'u': ['v', 'w'], 'v': ['f', 'u', 'w'],
                                      'w': ['f', 'v'],
                                      'x': ['s', 'z'], 'y': ['e', 'i'], 'z': ['c', 's']}

    # confusing letter + correct letter,可不可以根据迷惑字母的个数增加难度
    @property
    def letter_space(self):
        if self.difficulty_setting['confusing_letter_setting']:  # if it has confusing letters
            for correct_letter in self.available_letter:
                self.confusing_letter.append(random.choice(self._CONFUSING_LETTER_DIC[correct_letter]))
            self.available_letter += self.confusing_letter
        random.shuffle(self.available_letter)
        return self.available_letter

    def student_spelling(self, stu_feedback=None):
        """
               根据中文和音标拼写英语单词,学生拼写的单词应该在目标单词范围内，所以预测的目标结果，不应该是动作空间以外的字母
               如果我中文和音标输入，只输入中文，只输入英文会不会导致结果有很大的变化？
               :return: student spelling str
               """
        self.stu_feedback = stu_feedback
        chinese_phonetic_index = data_process(self.chinese_phonetic)
        chinese_phonetic_index_iter = DataLoader(chinese_phonetic_index, batch_size=1, collate_fn=generate_batch)
        self.stu_spelling, self.masks = evaluate(model, chinese_phonetic_index_iter, self.available_letter,
                                                 self.stu_feedback, self.masks, self.target_length + 1)

        return self.stu_spelling


class ExaminerPlayer(ExaminerInterface):
    def __init__(self, player_id, player_name):
        super().__init__(player_id, player_name)  # inherit abstract

    def give_feedback(self, stu_spelling, correct_spelling):  # 'n i r o ', 'i r o n'
        stu_spelling = stu_spelling.strip().split(' ')  # 首先将输入按照空格分割然后返回一个列表
        correct_spelling = correct_spelling.strip().split(' ')  # 首先将输入按照空格分割然后返回一个列表
        self.student_feedback = {}  # 每次给反馈清空以前的反馈
        # get the students' feedback
        for index in range(len(stu_spelling)):  # 通过索引来给字母打分
            current_letter = stu_spelling[index]  # get the letter
            if current_letter not in correct_spelling:  # 如果这个字母不在正确的字母中
                self.student_feedback[current_letter + '_' + str(index)] = 0  # 0 present red letter
            elif current_letter == correct_spelling[index]:
                self.student_feedback[current_letter + '_' + str(index)] = 2  # 2 present green letter
            else:
                self.student_feedback[current_letter + '_' + str(index)] = 1  # 1 present yellow letter
        # get the tutors' feedback
        student_spelling_completeness = 1 - Levenshtein.distance(''.join(stu_spelling),
                                                                 ''.join(correct_spelling)) / len(
            correct_spelling)
        student_spelling_accuracy = Levenshtein.ratio(''.join(stu_spelling), ''.join(correct_spelling))
        self.tutor_feedback = [round(student_spelling_completeness, 3),
                               round(student_spelling_accuracy, 3)]

        return self.student_feedback, self.tutor_feedback

