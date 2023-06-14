"""
define the interactive environment
"""
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
import random
from Word_Maker_RL.agents_interface import *
from student_spelling import evaluate, model, data_process, generate_batch
import Levenshtein as Levenshtein

# define the tasks pool
tasks_pool = {'人的 h j u m ʌ n': 'h u m a n', '谦逊的 h ʌ m b ʌ l': 'h u m b l e', '湿的 h j u m ʌ d': 'h u m i d',
              '墨水 ɪ ŋ k': 'i n k', '铁 aɪ ɝ n': 'i r o n', '语言 l æ ŋ ɡ w ʌ dʒ': 'l a n g u a g e',
              '洗衣房 l ɔ n d r i': 'l a u n d r y', '难题 p ʌ z ʌ l ': 'p u z z l e'}
# tasks_pool = {'语言 l æ ŋ ɡ w ʌ dʒ': 'l a n g u a g e'}


# 对于机会玩家，只要选任务就可以，其他的都在抽象类中实现了
class ChancePlayer(ChanceInterface):
    def __init__(self, custom_tasks_pool, maximum_game_rounds):  # initialize parameters
        super(ChancePlayer, self).__init__(custom_tasks_pool, maximum_game_rounds)  # inherit parent properties

    def select_word(self) -> Tuple[str, str]:
        self.ch_pho, self.word = random.choice(list(self.tasks_pool.items()))
        return self.ch_pho, self.word


chance_player = ChancePlayer(tasks_pool, 4)
chance_player_legal_tasks = chance_player.legal_tasks
print('all available tasks:', chance_player_legal_tasks)
chinese, word = chance_player.select_word()  # get the task word
print('the chinese_phonetic:', chinese)
print('the word:', word)
print('game state:', chance_player.is_terminal(2))


# 给难度，定义难度，返回难度参数， apply actions
class TutorPlayer(TutorInterface):
    def __init__(self):
        super(TutorPlayer, self).__init__()
        # question:如何设置为有无中文和音标
        self.difficulty_level_definition: Dict[int, Dict[str, int]] = {
            1: {'attempts': 4, 'confusing_letter_setting': 0, 'chinese_setting': 1, 'phonetic_setting': 1},
            2: {'attempts': 3, 'confusing_letter_setting': 1, 'chinese_setting': 1, 'phonetic_setting': 1},
            3: {'attempts': 2, 'confusing_letter_setting': 1, 'chinese_setting': 1, 'phonetic_setting': 0},
            4: {'attempts': 1, 'confusing_letter_setting': 1, 'chinese_setting': 0, 'phonetic_setting': 1}}

    # get the list of difficulty level, the difficulty level should keep or upgrade
    def legal_difficulty_levels(self, previous_difficulty_level) -> List[int]:
        difficulty_levels = [index for index, value in self.difficulty_level_definition.items()]
        return difficulty_levels[(previous_difficulty_level - 1):]

    def decide_difficulty_level(self, current_game_round: int) -> Dict[str, int]:
        return self.difficulty_level_definition[current_game_round]


tutor_player = TutorPlayer()
tutor_player_legal_difficulty_levels = tutor_player.legal_difficulty_levels(3)
print('legal difficulty levels: ', tutor_player_legal_difficulty_levels)
difficulty_setting = tutor_player.decide_difficulty_level(2)
print('difficulty setting:', difficulty_setting)


class StudentPlayer(StudentInterface):
    # 这里必须有父类的positional arguments, 如果要加新的参数则在后面添加就是
    def __init__(self, chinese_phonetic, target_english, current_difficulty_setting):
        # 父类的参数和子类的参数是一样的，然后最重要的是也会继承父类的初始化方法，所以不用自己定义，当然也可以重写
        super(StudentPlayer, self).__init__(chinese_phonetic, target_english, current_difficulty_setting)

        self._CONFUSING_LETTER_DIC: Dict[str, List[str]] = {'a': ['e', 'i', 'o', 'u', 'y'], 'b': ['d', 'p', 'q', 't'],
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
    def letter_space(self) -> List[str]:
        # 根据字典随机选择一个迷惑字母
        if self.difficulty_setting['confusing_letter_setting']:  # if it has confusing letters
            for correct_letter in self.available_letter:
                self.confusing_letter.append(random.choice(self._CONFUSING_LETTER_DIC[correct_letter]))
            self.available_letter += self.confusing_letter
        random.shuffle(self.available_letter)
        return self.available_letter

    def student_spelling(self, student_feedback=None) -> str:
        """
               根据中文和音标拼写英语单词,学生拼写的单词应该在目标单词范围内，所以预测的目标结果，不应该是动作空间以外的字母
               :return: student spelling str
               """

        chinese_phonetic_index = data_process(self.chinese_phonetic)
        chinese_phonetic_index_iter = DataLoader(chinese_phonetic_index, batch_size=1, collate_fn=generate_batch)
        student_spelling, masks = evaluate(model, chinese_phonetic_index_iter, self.available_letter,
                                                student_feedback, self.masks, self.target_length + 1)
        self.masks = masks  # 将记忆保存下来

        return student_spelling


# 统计除了空格以外的字符串的长度，+2的目的是因为预测的时候有首尾
student_player = StudentPlayer(chinese, word, difficulty_setting)
print('students letter space', student_player.letter_space())
student_spelling = student_player.student_spelling()
print('student spelling is:', student_spelling)


class ExaminerPlayer(ExaminerInterface):
    def __init__(self):
        super(ExaminerPlayer, self).__init__()  # inherit abstract

    def give_feedback(self, student_spelling: str, correct_spelling: str):  # 'n i r o ', 'i r o n'
        student_spelling: List[str] = student_spelling.strip().split(' ')  # 首先将输入按照空格分割然后返回一个列表
        correct_spelling: List[str] = correct_spelling.strip().split(' ')  # 首先将输入按照空格分割然后返回一个列表
        self.student_feedback: Dict[str, int] = {}  # 每次给反馈清空以前的反馈
        # get the students' feedback
        for index in range(len(student_spelling)):  # 通过索引来给字母打分
            current_letter = student_spelling[index]  # get the letter
            if current_letter not in correct_spelling:  # 如果这个字母不在正确的字母中
                self.student_feedback[current_letter + '_' + str(index)] = 0  # 0 present red letter
            elif current_letter == correct_spelling[index]:
                self.student_feedback[current_letter + '_' + str(index)] = 2  # 2 present green letter
            else:
                self.student_feedback[current_letter + '_' + str(index)] = 1  # 1 present yellow letter
        # get the tutors' feedback
        student_spelling_completeness: float = 1 - Levenshtein.distance(''.join(student_spelling),
                                                                 ''.join(correct_spelling)) / len(correct_spelling)
        student_spelling_accuracy: float = Levenshtein.ratio(''.join(student_spelling), ''.join(correct_spelling))
        self.tutor_feedback: List[float] = [round(student_spelling_completeness, 3), round(student_spelling_accuracy, 3)]

        return self.student_feedback, self.tutor_feedback


examiner_player = ExaminerPlayer()
student_feedback, tutor_feedback = examiner_player.give_feedback(student_spelling, word)
print(f'student feedback:{student_feedback},tutor feedback:{tutor_feedback}')

for i in range(10):
    student_spelling = student_player.student_spelling(student_feedback)
    student_feedback, tutor_feedback = examiner_player.give_feedback(student_spelling, word)
    print('student spelling is:', student_spelling)
    print(f'student feedback:{student_feedback},tutor feedback:{tutor_feedback}')
