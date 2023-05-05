"""
please ignore all the chinese characters that is for informing me something.
# 想一下如何利用学生的准确性和完整度
# 学生不知道正确答案是什么
# 我的payoff是什么？才能达到我的目的
problem 1： student do not forget anything. currently, if student get correct, all get correct in all levels.
"""

import random
from typing import List, Dict, Tuple, Union
import Levenshtein
import numpy as np
import pyspiel
from random import choice
import copy

# the tasks pool, tutor agent randomly select task from here
_TASKS_POOL: List[str] = ['bow', 'sink', 'crane', 'tenant', 'condemn', 'expedite']

# every letter have confusing letter list that used to confuse student.
_CONFUSING_LETTER_DIC: Dict[str, List[str]] = {'a': ['e', 'i', 'o', 'u', 'y'], 'b': ['d', 'p', 'q', 't'],
                                               'c': ['k', 's', 't', 'z'], 'd': ['b', 'p', 'q', 't'],
                                               'e': ['a', 'o', 'i', 'u', 'y'], 'f': ['v', 'w'], 'g': ['h', 'j'],
                                               'h': ['m', 'n'], 'i': ['a', 'e', 'o', 'y'], 'j': ['g', 'i'],
                                               'k': ['c', 'g'], 'l': ['i', 'r'], 'm': ['h', 'n'], 'n': ['h', 'm'],
                                               'o': ['a', 'e', 'i', 'u', 'y'], 'p': ['b', 'd', 'q', 't'],
                                               'q': ['b', 'd', 'p', 't'], 'r': ['l', 'v'], 's': ['c', 'z'],
                                               't': ['c', 'd'], 'u': ['v', 'w'], 'v': ['f', 'u', 'w'], 'w': ['f', 'v'],
                                               'x': ['s', 'z'], 'y': ['e', 'i'], 'z': ['c', 's']}


# chance agent randomly select a task word from tasks pool
class ChanceAction:
    def __init__(self, tasks_pool: List[str]) -> None:  # tasks pool list
        self._tasks_pool: List[str] = tasks_pool  # initial the task list
        self._task_word: str = random.choice(self._tasks_pool)  # randomly select a task

    @property
    def get_task_word(self) -> str:  # the action of chance agent will select a task word
        return self._task_word


# tutor action
class TutorAction:
    def __init__(self):
        self.letters_pool: List[str] = []  # tutor deal letters for student agent
        self.confusing_letter_pool: List[str] = []  # store confusing letter

    # input: confusing letters,  output: available letter for student agent
    def deal_letters(self, task_word: str, confuse_letters: int = 0) -> Tuple[List[str], int]:
        if confuse_letters:  # with confusing letter
            self.letters_pool: List[str] = [correct_letter for correct_letter in task_word]  # get correct letters
            for letter in self.letters_pool:
                self.confusing_letter_pool.append(random.choice(_CONFUSING_LETTER_DIC[letter])) # select confusing letter
                # correct letter + confusing letter
            self.letters_pool: List[str] = self.letters_pool + self.confusing_letter_pool
        else:  # without confusing letter
            self.letters_pool: List[str] = [letter for letter in task_word]  # only get the correct letter
        random.shuffle(self.letters_pool)  # shuffle the order of letter
        #  return the available letter for student agent, the length of word
        return self.letters_pool, len(task_word)

    # tutor agent decide which difficulty level to deliver
    def adjust_difficulty_level(self, difficulty: int) -> Tuple[int, int]:
        difficult_level: Dict[int, Dict[str, int]] = {1: {'attempts': 4, 'confusing_letter': 0},
                                                      2: {'attempts': 3, 'confusing_letter': 1},
                                                      3: {'attempts': 2, 'confusing_letter': 1},
                                                      4: {'attempts': 1, 'confusing_letter': 1}}
        # return the [total number of attempts students get in each round, confusing letter setting]
        return difficult_level[difficulty]['attempts'], difficult_level[difficulty]['confusing_letter']


# examiner player
class ExaminerAction:
    def __init__(self):
        self.red: Dict[str, str] = {}  # record wrong letter {correct_letter:[position,confusing_letter]}
        self.yellow: Dict[str, str] = {}  # record yellow letter {yellow_letter:{position}}
        self.green: Dict[str, str] = {}  # record green letter {green_letter:{position}}
        self.completeness: float = 0.  # completeness of student agent answer
        self.accuracy: float = 0.  # accuracy of student agent answer

    # input： student agent spelling, output: completeness, accuracy, feedback information
    def check_spelling(self, student_agent_spelling: str, answer: str) -> Tuple[
        Dict[str, str], Dict[str, str], Dict[str, str], float, float]:
        for index in range(len(answer)):  #
            if student_agent_spelling[index] not in answer:  # the letter not in correct letter labeled red
                self.red[answer[index] + '_' + str(index)] = student_agent_spelling[index]
            elif student_agent_spelling[index] == answer[index]:  # correct at the correct position labeled green
                self.green[answer[index] + '_' + str(index)] = student_agent_spelling[index]
            else:  # correct letter at the wrong place labeled yellow
                self.yellow[student_agent_spelling[index] + '_' + str(index)] = answer[index]
        self.accuracy = Levenshtein.ratio(student_agent_spelling, answer)  # evaluate accuracy
        self.completeness = 1 - Levenshtein.distance(student_agent_spelling, answer) / len(
            answer)  # evaluate completeness
        return self.red, self.green, self.yellow, self.accuracy, self.completeness


class StudentAction:
    def __init__(self, available_letters: List[str], word_length: int):
        self._available_letters: List[str] = available_letters  # available letters for student agent
        self.student_spelling: str = ''  # store student spelling
        self.word_length: int = word_length  # word length
        self.memory: Dict[str:str] = {}  # student agent memory
        self.available_index: List[int] = []  # corresponding to the available letter
        self.available_letters_copy: List[int] = []  # copy the available letter

    # learn from previous mistake
    def learn_from_history(self, red_letters: Dict[str, str], green_letters: Dict[str, str],
                           yellow_letters: Dict[str, str], spelling_accuracy: float,
                           spelling_completeness: float) -> str:
        self.available_index = [i for i in range(self.word_length)]
        # for red letter, permanent remove red letters
        for _, values in red_letters.items():  # red letter
            self._available_letters.remove(values[0])  # red letter remove from available letters
        self.available_letters_copy: List[str] = copy.deepcopy(
            self._available_letters)  # deepcopy the available letters

        # for green letters, take up index and letter
        for key, values in green_letters.items():
            green_letter, green_letter_index = key.split('_')  #
            self.memory[int(green_letter_index)] = green_letter  # [index:letter]
            self.available_index.remove(int(green_letter_index))
            self.available_letters_copy.remove(green_letter)

        # yellow letter，
        if yellow_letters:
            yellow_letter_index: List[str] = list(yellow_letters.keys())
            for item in yellow_letter_index:  # get the index letter
                yellow_available_index: List[int] = copy.deepcopy(self.available_index)  # copy the available index letter
                letter_index: int = int(item.split('_')[1])  # get the previous letter
                # yellow index cannot take up the index
                if letter_index in self.available_index and letter_index != self.available_index[0]:
                    yellow_available_index.remove(letter_index)  # 得把当前的索引移除了，从剩下的索引中选择
                _current_index: int = random.choice(yellow_available_index)
                self.memory[_current_index] = item.split('_')[0]  # [index, letter]
                self.available_letters_copy.remove(item.split('_')[0])
                self.available_index.remove(_current_index)

        # for the rest of letters, randomly select from remaining available letters
        for index in self.available_index:  # the remaining available position
            chosen_letter: str = random.choice(self.available_letters_copy)
            self.memory[index] = chosen_letter  # [index:letter]
            self.available_letters_copy.remove(chosen_letter)  #
        list_tuple_spelling: List[str, str] = sorted(self.memory.items(), key=lambda memory_item: memory_item[0])
        self.student_spelling = ''
        for index, letter in list_tuple_spelling:
            self.student_spelling += letter  # student new spelling
        # return student spelling
        return self.student_spelling

    def spell_word(self) -> str:  # spell word, randomly spelling
        # first spelling
        letter_indexes: List[int] = [index for index in range(len(self._available_letters))]  # get all index
        while len(self.student_spelling) < self.word_length:
            chosen_index: int = random.choice(letter_indexes)  # randomly select an index
            chosen_letter: str = self._available_letters[chosen_index]
            self.student_spelling += chosen_letter
            letter_indexes.pop(letter_indexes.index(chosen_index))
        # return student spelling
        return self.student_spelling

# terminal player
class TerminalAction:
    STOP: bool = True


# index number to player
PLAYERS: Dict[int, any] = {-1: 'chance_player', -4: 'terminal_player', 1: 'tutor_player',
                           2: 'student_player', 3: 'examiner_player'}

# index number to agent player
_ACTION_SPACE: Dict[int, List[str]] = {-1: ['get_task_word'], 1: ['deal_letters', 'adjust_difficulty_level'],
                                       2: ['spell_word', 'learn_from_history'],
                                       3: ['check_spelling'], -4: ["STOP"]}

_NUM_PLAYERS = 3  # student agent, tutor agent，examiner agent
_GAME_TYPE = pyspiel.GameType(
    short_name="word_maker",  # game name
    long_name="word maker",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,  # sequential game
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,  # no chance node?--------
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,  # perfect information
    utility=pyspiel.GameType.Utility.IDENTICAL,  # GENERAL_SUM game---------
    reward_model=pyspiel.GameType.RewardModel.REWARDS,  # rewards during gameplay
    max_num_players=_NUM_PLAYERS,  # maximum number of players
    min_num_players=_NUM_PLAYERS,  # minimum number of players
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=6,  # student action+teacher action+examiner action
    max_chance_outcomes=6,  # the length of tasks pool
    num_players=_NUM_PLAYERS,  # number of players
    min_utility=-2.0,  # minimum score?------------
    max_utility=2.0,  # maximum score?---------------
    # utility_sum=None,  # identical
    max_game_length=4)  # round_1,round_2,round_3,round_4?----------------


# class WordMakerGame(pyspiel.Game):
#     """A Python version of Kuhn poker."""
#
#     def __init__(self, params=None):
#         super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
#
#     def new_initial_state(self):  # 返回游戏最开始的状态，并且可以通过返回的类调用state所有想知道的值
#         """Returns a state corresponding to the start of a game."""
#         return WordMakerState(self)

# def make_py_observer(self, iig_obs_type=None, params=None):
#     """Returns an object used for observing game state."""
#     return WordMakerObserver(iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), params)


class WordMakerState:

    def __init__(self, tasks_pool: List[str]):
        self.chance_player = None
        self.terminal_player = None
        self.tutor_player = None
        self.student_player = None
        self.examiner_player = None
        self._TASKS_POOL: List[str] = tasks_pool  # get the task pool
        self._current_task: str = ''  # the current task
        self.task_word_length: int = 0  # the word length
        self._game_over: bool = False  # decide game is over or not
        self.student_available_letter: List[str] = []  # available letter student get
        self._student_agent_spelling: str = ''  # get student spelling
        self.history: Dict = {}  # record the history state information
        self._next_player: int = -1  # the initial first player is chance player
        self.current_difficulty_level: int = 1  # initial difficulty level
        self.total_attempts: int = 0  #  the total numbers of attempts students get
        self.confusing_letter_setting: int = 0  # confusing letter setting
        self.student_answer_feedback: Tuple[any] = ()
        self.current_game_round: int = 1  # initialize the game round
        self.student_answer_accuracy: float = 0.  # record student spelling
        self.player_legal_action: str = ''  # record player legal action
        self.difficulty_level_decided: bool = False  # decide difficulty
        self.next_attempt: int = 0  # record the next students attempt

    # get the legal action according to the current player
    def legal_actions(self, player: int) -> str:  # get the legal action
        if player == -1:  # chance player
            self.player_legal_action = 'get_task_word'
        elif player == -4:  # terminal player
            self.player_legal_action = "STOP"
        elif player == 1 and self.difficulty_level_decided:  # tutor player
            self.player_legal_action = 'deal_letters'
        elif player == 1 and not self.difficulty_level_decided:  # tutor player
            self.player_legal_action = 'adjust_difficulty_level'
        elif player == 2 and self.next_attempt == 0 and self.current_game_round == 1:  # student agent
            self.player_legal_action = 'spell_word'
        elif player == 2:  # student agent
            self.player_legal_action = 'learn_from_history'
        elif player == 3:  # examiner agent
            self.player_legal_action = 'check_spelling'
        return self.player_legal_action

    #  return the current player
    def current_player(self) -> int:
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            # -4 represents terminal player
            return -4
        elif not self._current_task:  # if current task is empty, return chance player
            # -1 represents chance player
            return -1
        else:  # otherwise, other player
            return self._next_player

    # 要加切换玩家的动作
    def apply_action(self, action):  # input:action
        # get the current player
        current_player_id: List[int] = [index for index, values in _ACTION_SPACE.items() if action in values]
        if current_player_id[0] == -1:  # if chance player
            if self.current_game_round == 1:  # select task word only in first game round
                self.chance_player = ChanceAction(self._TASKS_POOL)  # instance
                self._current_task: str = getattr(self.chance_player, action)  # student agent select a word
                self._next_player = 1  # after select task, the next player will be tutor player

        elif current_player_id[0] == -4:  # if terminal player，only used for terminate
            self.terminal_player = TerminalAction()  # instance
            self._game_over: str = getattr(self.terminal_player, action)  # game over
            self._next_player = -1  # after terminating, chance player need to choose a task word

        elif current_player_id[0] == 1:  # if tutor player
            self.tutor_player = TutorAction()  # instance
            if action == 'deal_letters':
                self.student_available_letter, self.task_word_length = getattr(self.tutor_player, action) \
                    (self._current_task, self.confusing_letter_setting)
                self._next_player = 2  # after deal letter, student agent spell word
            else:
                self.total_attempts, self.confusing_letter_setting = getattr(self.tutor_player, action) \
                    (self.current_difficulty_level)
                self._next_player = 1  # decide difficulty level, then tutor agent deal letters
                self.difficulty_level_decided = True  # difficulty settled

        elif current_player_id[0] == 2:  # if student player
            self.student_player = StudentAction(self.student_available_letter, self.task_word_length)  # instance
            if action == 'spell_word':  # randomly select letter
                self._student_agent_spelling = getattr(self.student_player, action)()
            else:
                # learn from history
                self._student_agent_spelling = getattr(self.student_player, action)(*self.student_answer_feedback)
            self.next_attempt += 1  # student attempt plus one
            self._next_player = 3  # after spelling, next players will be the examiner

        else:  # if examiner player
            self.examiner_player = ExaminerAction()  # instance
            # get the feedback (red, yellow, green, accuracy, completeness)
            self.student_answer_feedback = getattr(self.examiner_player, action)(self._student_agent_spelling,
                                                                                 self._current_task)
            _, _, _, self.student_answer_accuracy, _ = self.student_answer_feedback
            if self.student_answer_accuracy != 1.0:  # spelling not correct
                if self.next_attempt < self.total_attempts:  # run out of total attempts?
                    self._next_player = 2  # student continue to spelling

                else:  # run out of total attempts
                    if self.current_game_round < 4:  # game over or not?
                        self.current_game_round += 1  # game round plus one
                        self.current_difficulty_level += 1  # difficulty plus one
                        self.difficulty_level_decided = False  #
                        self._next_player = 1  # tutor agent
                        self.next_attempt = 0  # initial attempt
                    else:  # the game round end
                        self.current_game_round = 1
                        self._next_player = -4

            else:  # correct spelling
                if self.current_game_round < 4:  # do not reach round 4
                    self.current_difficulty_level += 1  # difficulty plus one
                    self.difficulty_level_decided = False
                    self.current_game_round += 1  # game round plus one
                    self._next_player = 1  # the next player tutor agent to decide difficulty
                    self.next_attempt = 0  # initial attempt

                else:  # 如果是最后一轮答对的
                    self.current_game_round = 1
                    self._next_player = -4

    def is_terminal(self):  # want to know the game terminate or not
        return self._game_over

    def rewards(self):  # the rewards function is
        """Total reward for each player over the course of the game so far."""
        pass

    # print information
    def __str__(self):
        return f"player_action:{self.player_legal_action},next_player:{PLAYERS[self._next_player]}, current_task:{self._current_task},task_word_length:{self.task_word_length}," \
               f"game_state:{self._game_over},current_difficulty_level:{self.current_difficulty_level}, current_round:{self.current_game_round}," \
               f"total_attempts:{self.total_attempts},confusing_letter_set:{self.confusing_letter_setting}," \
               f"student_available_letter:{self.student_available_letter},student_spelling:{self._student_agent_spelling}," \
               f"next_attempts:{self.next_attempt},student_answer_feedback:{self.student_answer_feedback}," \
               f" "

# iterate information
for i in range(10):
    print(i)
    state = WordMakerState(tasks_pool=_TASKS_POOL)
    while not state.is_terminal():
        state.apply_action(state.legal_actions(state._next_player))
        print(state)


# ------------------------the below are observation information, you can ignore below information--------------------
_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']


class WordMakerObserver:
    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # Determine which observation pieces we want to include.
        # chance, tutor, student, examiner
        pieces = [("player", 4, (4,))]  # the format of one-hot
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:  # 正确的答案
            pieces.append(("correct_answer", 6, (6,)))  # indicate the index of tasks list
            if iig_obs_type.public_info:  # public information
                # perfect recall means it will record spelling history
                if iig_obs_type.perfect_recall:  # one-hot的形式记录决策路径，最多有三个回合，所以需要三行，左边代表P，右边代表B
        #                 pieces.append(("betting", 6, (3, 2)))
        #             else:  # 如果不允许回忆之前的路径，就看当前的资金池的信息
        #                 pieces.append(("pot_contribution", 2, (2,)))
        #
        #         # Build the single flat tensor. 看需要多少列才能表达了所有的信息
        #         total_size = sum(size for name, size, shape in pieces)
        #         self.tensor = np.zeros(total_size, np.float32)  # 那么一个tensor就包括想要的所有信息
        #
        #         # Build the named & reshaped views of the bits of the flat tensor.
        #         # 重新搭建之前坦平的tensor
        self.dict = {}  # 定义一个字典来记录玩家看到的obs

    #         index = 0  # 记录一个索引，用来隔断各个信息
    #         for name, size, shape in pieces:  # 循环所有的obs，用字典的形式记录所有的信息
    #             self.dict[name] = self.tensor[index:index + size].reshape(shape)
    #             index += size
    #
    #     def set_from(self, state, player):  # 该方式返回的是字典的形式
    #         """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    #         self.tensor.fill(0)  # 首先tensor中的所有值都是0
    #         if "player" in self.dict:
    #             self.dict["player"][player] = 1  # 用【0，1】【1，0】表示哪个玩家
    #         if "private_card" in self.dict and len(state.cards) > player:  # 玩家手里得有牌
    #             self.dict["private_card"][state.cards[player]] = 1  # 【1,0,0】[0,1,0][0,0,1] 分别表示哪张牌
    #         if "pot_contribution" in self.dict:  # 想要看到赌注
    #             self.dict["pot_contribution"][:] = state.pot  # 直接复制，不用one-hot
    #         if "betting" in self.dict:  # 看看有没有押注
    #             for turn, action in enumerate(state.bets):  # 返回（索引，值）
    #                 self.dict["betting"][turn, action] = 1  # 返回的是进行的什么操作

    def string_from(self, state, player):  # 返回string的形式，更加符合人类的语言
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        if "player" in self.dict:
            pieces.append(f"p{player}")  # p0, p1
        if "private_card" in self.dict and len(state.cards) > player:
            pieces.append(f"card:{state.cards[player]}")  # card:0,card:1,card:2
        if "pot_contribution" in self.dict:
            pieces.append(f"pot[{int(state.pot[0])} {int(state.pot[1])}]")  # pot[2 2]
        if "betting" in self.dict and state.bets:
            pieces.append("".join("pb"[b] for b in state.bets))  # 'pb'
        return " ".join(str(p) for p in pieces)

# pyspiel.register_game(_GAME_TYPE, WordMakerGame)  # 注册游戏，然后很多方法就可以直接使用了
