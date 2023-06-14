from state_interface import StateInterface, _PLAYER_ACTION
from typing import List, Tuple, Dict
from agents_instance import ChancePlayer, TutorPlayer, StudentPlayer, ExaminerPlayer


class WordMakerState(StateInterface):
    def __init__(self, tasks: Dict[str, str], total_game_round: int = 4):
        super().__init__(tasks, total_game_round)
        self._chance_player = ChancePlayer(self._tasks_pool)  # initialize chance player
        self._task_ch_pho = ''
        self._task_spelling = ''
        self._tutor_player = TutorPlayer()  # initialize tutor player
        self._current_difficulty_setting = self._tutor_player.current_difficulty_level
        self._student_player = StudentPlayer(self._task_ch_pho, self._task_spelling, self._current_difficulty_setting)
        self._student_spelling: str = ''
        self._examiner_player = ExaminerPlayer()
        self._student_feedback = {}
        self._tutor_feedback = []

    def apply_action(self, action: str) -> _PLAYER_ACTION:
        # chance -> teacher -> student -> examiner -> not sure
        if action == 'select_word':
            self._task_ch_pho, self._task_spelling = self._chance_player.select_word
            self._player_action = _PLAYER_ACTION('tutor', 'decide_difficulty_level')
        elif action == 'decide_difficulty_level':
            self._current_difficulty_setting = self._tutor_player.decide_difficulty_level(self._current_game_round)
            self._player_action = _PLAYER_ACTION('student', 'student_spelling')
        elif action == 'student_spelling':
            self._student_player = StudentPlayer(self._task_ch_pho, self._task_spelling,
                                                 self._current_difficulty_setting)  # reinitialize student player
            self._student_spelling = self._student_player.student_spelling(self._student_feedback)
            self._player_action = _PLAYER_ACTION('examiner', 'give_feedback')
        elif action == 'give_feedback':
            self._student_feedback, self._tutor_feedback = self._examiner_player.give_feedback(self._student_spelling,
                                                                                               self._task_spelling)
            self._player_action = _PLAYER_ACTION('chance', 'select_word')
        return self._player_action

    @property
    def current_difficulty_setting(self):
        return self._current_difficulty_setting

    def is_terminal(self) -> bool:
        """Returns True if the game is over."""
        return self._game_over
