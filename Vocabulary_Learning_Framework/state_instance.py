import Levenshtein

from state_interface import StateInterface


class VocabSpellState(StateInterface):
    def __init__(self, vocab_sessions):
        super().__init__(vocab_sessions)

    def legal_actions(self, player_ID):
        return self._legal_actions[player_ID]

    def tutor_legal_actions(self):
        """define the tutor action, which is the lengths of words in one session """
        for task in self._vocab_session:
            self._legal_actions[1].append(len(''.join(task[1].split())))

    def spilt_task(self, action):
        """get the 'condition' and word_length for student, and answer for examiner """
        for task in self._vocab_session:
            if len(''.join(task[1].split())) == action:
                self._condition = task[0]
                self._answer = task[1]
                self._answer_length = action

    def student_spelling(self, actions):
        """ convert index to letter """
        self._stu_spelling = [self._LETTERS[action] for action in actions]

    def acc_com(self):
        """ calculate student spelling' completeness and accuracy """
        self._completeness = round(1 - Levenshtein.distance(''.join(self._stu_spelling),
                                                            ''.join(self._answer.split(' '))) / self._answer_length, 2)
        self._accuracy = round(Levenshtein.ratio(''.join(self._stu_spelling), ''.join(self._answer.split(' '))), 2)

    def apply_action(self, action):
        if self._current_player == 0:
            self._legal_actions[self._current_player].remove(action)
            self._vocab_session = self._vocab_sessions[action]
            self.tutor_legal_actions()  # define the second player legal actions
        elif self._current_player == 1:
            self._legal_actions[self._current_player].remove(action)  # remover one of the length
            self.spilt_task(action)
        elif self._current_player == 2:
            self.student_spelling(action)
        elif self._current_player == 3:
            self._letter_feedback = action
            self.acc_com()
        self._current_player += 1
