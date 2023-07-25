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
                 vocabulary_data,
                 new_words_number,
                 new_selection_method):
        super().__init__(player_id,
                         player_name,
                         vocabulary_data,
                         new_words_number,
                         new_selection_method)

    def random_method(self):
        """the random method of new words"""
        random.shuffle(self._vocabulary_data)
        for i in range(0, len(self._vocabulary_data), self._new_words_number):
            vocabulary_pieces = self._vocabulary_data[i:i + self._new_words_number]
            self._vocabulary_session.append(vocabulary_pieces)

    def piece_data(self):
        if self._new_selection_method == 'sequential':
            self.sequential_method()
        elif self._new_selection_method == 'random':
            self.random_method()

    def session_collector(self, current_session) -> List[List[str]]:
        """
                    :return: one session words data over time
                    """
        return self._vocabulary_session[current_session]


# PresentWord Agent
class PresentWordPlayer(PresentWordInterface):
    def __init__(self,
                 player_id,
                 player_name,
                 session_data,
                 selection_method):
        super().__init__(player_id,
                         player_name,
                         session_data,
                         selection_method)


# # tutor player
# class TutorPlayer(TutorInterface):
#     def __init__(self, player_id, player_name, tasks_pair):
#         super().__init__(player_id, player_name, tasks_pair)
#         self.difficulty_level_definition = {
#             1: {'attempts': 4, 'confusing_letter_setting': 0, 'chinese_setting': 1, 'phonetic_setting': 1},
#             2: {'attempts': 3, 'confusing_letter_setting': 1, 'chinese_setting': 1, 'phonetic_setting': 1},
#             3: {'attempts': 2, 'confusing_letter_setting': 1, 'chinese_setting': 1, 'phonetic_setting': 0},
#             4: {'attempts': 1, 'confusing_letter_setting': 1, 'chinese_setting': 0, 'phonetic_setting': 1}}
#
#     # get the list of difficulty level, the difficulty level should keep or upgrade
#     def legal_difficulty_levels(self):
#
#         difficulty_levels = [index for index, value in self.difficulty_level_definition.items()]
#         self.legal_difficulty_level = difficulty_levels[(previous_difficulty_level - 1):]
#
#         return self.legal_difficulty_level
#
#     # get the difficulty setting [self,state] -> policy->action
#     # state->[accuracy,completeness,difficulty level,current game round,red,green,yellow]
#     def decide_difficulty_level(self, current_game_round):
#         return self.difficulty_level_definition[current_game_round]
#
#
# # student player
# class StudentPlayer(StudentInterface):
#     def __init__(self, player_id, player_name, chinese_phonetic, target_english, current_difficulty_setting):
#         super().__init__(player_id, player_name, chinese_phonetic, target_english, current_difficulty_setting)
#
#         self._CONFUSING_LETTER_DIC = {'a': ['e', 'i', 'o', 'u', 'y'], 'b': ['d', 'p', 'q', 't'],
#                                       'c': ['k', 's', 't', 'z'], 'd': ['b', 'p', 'q', 't'],
#                                       'e': ['a', 'o', 'i', 'u', 'y'], 'f': ['v', 'w'],
#                                       'g': ['h', 'j'],
#                                       'h': ['m', 'n'], 'i': ['a', 'e', 'o', 'y'], 'j': ['g', 'i'],
#                                       'k': ['c', 'g'], 'l': ['i', 'r'], 'm': ['h', 'n'],
#                                       'n': ['h', 'm'],
#                                       'o': ['a', 'e', 'i', 'u', 'y'], 'p': ['b', 'd', 'q', 't'],
#                                       'q': ['b', 'd', 'p', 't'], 'r': ['l', 'v'], 's': ['c', 'z'],
#                                       't': ['c', 'd'], 'u': ['v', 'w'], 'v': ['f', 'u', 'w'],
#                                       'w': ['f', 'v'],
#                                       'x': ['s', 'z'], 'y': ['e', 'i'], 'z': ['c', 's']}
#
#     # confusing letter + correct letter,可不可以根据迷惑字母的个数增加难度
#     @property
#     def letter_space(self):
#         if self.difficulty_setting['confusing_letter_setting']:  # if it has confusing letters
#             for correct_letter in self.available_letter:
#                 self.confusing_letter.append(random.choice(self._CONFUSING_LETTER_DIC[correct_letter]))
#             self.available_letter += self.confusing_letter
#         random.shuffle(self.available_letter)
#         return self.available_letter
#
#     def student_spelling(self, stu_feedback=None):
#         """
#                根据中文和音标拼写英语单词,学生拼写的单词应该在目标单词范围内，所以预测的目标结果，不应该是动作空间以外的字母
#                如果我中文和音标输入，只输入中文，只输入英文会不会导致结果有很大的变化？
#                :return: student spelling str
#                """
#         self.chinese_phonetic_index = data_process(self.chinese_phonetic)
#         if self.difficulty_setting['phonetic_setting'] == 0:  # only input chinese
#             self.chinese_phonetic_index = [self.chinese_phonetic_index[0][:1]]
#         if self.difficulty_setting['chinese_setting'] == 0:  # only input phonetic
#             self.chinese_phonetic_index = [self.chinese_phonetic_index[0][1:]]
#
#         self.chinese_phonetic_index_iter = DataLoader(self.chinese_phonetic_index, batch_size=1,
#                                                       collate_fn=generate_batch)
#         self.stu_feedback = stu_feedback
#         self.stu_spelling, self.masks = evaluate(model, self.chinese_phonetic_index_iter, self.available_letter,
#                                                  self.stu_feedback, self.masks, self.target_length + 1)
#         return self.stu_spelling
#
#     def student_memorizing(self):
#         # train_student(self.chinese_phonetic_index_iter, self.stu_feedback, self.masks, self.target_length + 1, self.target_spelling)
#         train_student(dict({self.chinese_phonetic: self.target_spelling}))
#
#
# class ExaminerPlayer(ExaminerInterface):
#     def __init__(self, player_id, player_name):
#         super().__init__(player_id, player_name)  # inherit abstract
#
#     def give_feedback(self, stu_spelling, correct_spelling):  # 'n i r o ', 'i r o n'
#         stu_spelling = stu_spelling.strip().split(' ')  # 首先将输入按照空格分割然后返回一个列表
#         correct_spelling = correct_spelling.strip().split(' ')  # 首先将输入按照空格分割然后返回一个列表
#         self.student_feedback = {}  # 每次给反馈清空以前的反馈
#         # get the students' feedback
#         for index in range(len(stu_spelling)):  # 通过索引来给字母打分
#             current_letter = stu_spelling[index]  # get the letter
#             if current_letter not in correct_spelling:  # 如果这个字母不在正确的字母中
#                 self.student_feedback[current_letter + '_' + str(index)] = 0  # 0 present red letter
#             elif current_letter == correct_spelling[index]:
#                 self.student_feedback[current_letter + '_' + str(index)] = 2  # 2 present green letter
#             else:
#                 self.student_feedback[current_letter + '_' + str(index)] = 1  # 1 present yellow letter
#         # get the tutors' feedback
#         student_spelling_completeness = 1 - Levenshtein.distance(''.join(stu_spelling),
#                                                                  ''.join(correct_spelling)) / len(correct_spelling)
#         student_spelling_accuracy = Levenshtein.ratio(''.join(stu_spelling), ''.join(correct_spelling))
#         self.tutor_feedback = [round(student_spelling_completeness, 3),
#                                round(student_spelling_accuracy, 3)]  # red, green, green, current difficulty, current round
#
#         return self.student_feedback, self.tutor_feedback
