from state_interface import StateInterface
import numpy as np


class VocabSpellState(StateInterface):
    def __init__(self, vocab_sessions):
        super().__init__(vocab_sessions)

    def legal_actions(self, player_ID):
        """get the legal action of one agent"""
        return self._legal_actions[player_ID]

    def tutor_legal_actions(self):
        """define the tutor action, which is the words in one session """
        self._legal_actions[1] = self._vocab_session
        self._current_player += 1

    def spilt_task(self, action):
        """get the 'condition' and word_length for student, and answer for examiner """
        for task in self._vocab_session:
            if task == action:  # 如果两个任务完全相等
                self._current_corpus = tuple(task)  # the type of corpus is tuple('s ɛ n t ʌ n s', 's e n t e n c e')
                self._condition = task[0]
                self._answer = task[1]
                self._answer_length = len(''.join(task[1].split(' ')))
                break

    def student_spelling(self, actions):
        """ convert index to letter """
        self._stu_spelling = [self._LETTERS[action] for action in actions]
        self._current_player += 1

    # 保存历史记录
    def apply_action(self, action):
        if self._current_player == 0:  # 每个session结束以后，才到 player 0
            self._legal_actions[self._current_player].remove(action)  # 选择了就直接移除
            self._vocab_session = self._vocab_sessions[action]   #
            self.tutor_legal_actions()  # define the second player legal actions
        elif self._current_player == 1:  # tutor
            self.spilt_task(action)
            self._legal_actions[self._current_player].remove(action)  # remover the task
            self._current_player += 1
        elif self._current_player == 2:
            self.student_spelling(action)
        elif self._current_player == 3:
            self._examiner_feedback = action
            # store all information
            self._history[self._current_corpus] = self._examiner_feedback
            # if session task is empty, then select a new session, else continue to select new word from the session
            if len(self.legal_actions(1)) == 0:
                """如果当前的session为空则第一个玩家重新选择session"""
                self._current_player = 0
                self._current_session_num += 1
            else:
                self._current_player = 1
        if len(self._legal_actions[1]) == 0 and len(self._legal_actions[0]) == 0:
            """ if both the sessions list and session list are empty,then game over """
            self._game_over = True

    def reward_function(self, information):
        """the reward only for tutor agent, the information just accuracy
        奖励应该和准确度成正比，虽然和单词长度也有一定的关系"""
        scaled_accuracy = np.tanh(information * (np.pi / 2) - (np.pi / 4))
        return np.tan(scaled_accuracy)

