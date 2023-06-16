"""
for testing the agent interface and instance
1: agents instance
2: the method of tutor should be changed in the future
"""

from torch.utils.data import DataLoader
import random
from Word_Maker_RL.agents_interface import *
from student_spelling import evaluate, model, data_process, generate_batch
import Levenshtein as Levenshtein

# define the tasks pool
tasks_pool = {'人的 h j u m ʌ n': 'h u m a n', '谦逊的 h ʌ m b ʌ l': 'h u m b l e', '湿的 h j u m ʌ d': 'h u m i d',
              '墨水 ɪ ŋ k': 'i n k', '铁 aɪ ɝ n': 'i r o n', '语言 l æ ŋ ɡ w ʌ dʒ': 'l a n g u a g e',
              '洗衣房 l ɔ n d r i': 'l a u n d r y', '难题 p ʌ z ʌ l ': 'p u z z l e'}


class ChancePlayer(ChanceInterface):
    def __init__(self, player_id, player_name, custom_tasks_pool):
        super().__init__(player_id, player_name, custom_tasks_pool)

    @property
    def select_word(self):
        self._ch_pho, self._word = random.choice(list(self._tasks_pool.items()))
        return self._ch_pho, self._word


# chance_player = ChancePlayer(player_id=0, player_name='student', custom_tasks_pool=tasks_pool)
# chinese, word = chance_player.select_word
# print(chance_player.player_id)
# print(chance_player.player_name)
# print('the chinese_phonetic:', chinese)
# print('the word:', word)


class TutorPlayer(TutorInterface):
    def __init__(self, player_id, player_name):
        super().__init__(player_id, player_name)
        # question:如何设置为有无中文和音标
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


# tutor_player = TutorPlayer(player_id=1, player_name='tutor')
# tutor_player_legal_difficulty_levels = tutor_player.legal_difficulty_levels(3)
# print('legal difficulty levels: ', tutor_player_legal_difficulty_levels)
# difficulty_setting = tutor_player.decide_difficulty_level(1)
# print('difficulty setting:', difficulty_setting)


class StudentPlayer(StudentInterface):
    # 这里必须有父类的positional arguments, 如果要加新的参数则在后面添加就是
    def __init__(self, player_id, player_name, chinese_phonetic, target_english, current_difficulty_setting):
        # 父类的参数和子类的参数是一样的，然后最重要的是也会继承父类的初始化方法，所以不用自己定义，当然也可以重写
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
               :return: student spelling str
               """
        self.stu_feedback = stu_feedback
        chinese_phonetic_index = data_process(self.chinese_phonetic)
        chinese_phonetic_index_iter = DataLoader(chinese_phonetic_index, batch_size=1, collate_fn=generate_batch)
        self.stu_spelling, self.masks = evaluate(model, chinese_phonetic_index_iter, self.available_letter,
                                                 self.stu_feedback, self.masks, self.target_length + 1)

        return self.stu_spelling


# 统计除了空格以外的字符串的长度，+1的目的是因为预测的时候有首
# student_player = StudentPlayer(2, 'student', chinese, word, difficulty_setting)
# print('students letter space', student_player.letter_space)
# student_spelling = student_player.student_spelling()
# print('student spelling is:', student_spelling)


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

#
# examiner_player = ExaminerPlayer(3, 'examiner')
# student_feedback, tutor_feedback = examiner_player.give_feedback(student_spelling, word)
# print(f'student feedback:{student_feedback},tutor feedback:{tutor_feedback}')
#
# for i in range(10):
#     student_spelling = student_player.student_spelling(student_feedback)
#     student_feedback, tutor_feedback = examiner_player.give_feedback(student_spelling, word)
#     print('student spelling is:', student_spelling)
#     print(f'student feedback:{student_feedback},tutor feedback:{tutor_feedback}')
