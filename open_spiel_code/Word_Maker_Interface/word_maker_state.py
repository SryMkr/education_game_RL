from state_interface import StateInterface, _PLAYER_ACTION
from agents_instance import ChancePlayer, TutorPlayer, StudentPlayer, ExaminerPlayer
from typing import List, Tuple, Dict, Union, Optional
import threading

event = threading.Event()


class WordMakerState(StateInterface):
    def __init__(self, tasks, total_game_round):
        super().__init__(tasks, total_game_round)
        self._chance_player = ChancePlayer(player_id=0, player_name='chance',
                                           custom_tasks_pool=self._tasks_pool)  # initialize chance player
        self._task_ch_pho = ''
        self._task_spelling = ''

        self._tutor_player = TutorPlayer(player_id=1, player_name='tutor')  # initialize tutor player
        self._current_difficulty_setting = self._tutor_player.difficulty_level_definition[
            self._current_difficulty_level]
        self._current_attempt: int = 1
        self._total_attempt = self._current_difficulty_setting['attempts']

        self._student_player = StudentPlayer(player_id=2, player_name='student', chinese_phonetic=self._task_ch_pho,
                                             target_english=self._task_spelling,
                                             current_difficulty_setting=self._current_difficulty_setting)
        self._available_letter = self._student_player.letter_space
        self._student_spelling: str = ''

        self._examiner_player = ExaminerPlayer(player_id=3, player_name='examiner')
        self._student_feedback: Optional[Dict[str, int]] = {}
        self._tutor_feedback: List[float] = []
        self._difficulty_change: bool = True

    def apply_action(self, action: str) -> _PLAYER_ACTION:
        # chance -> teacher -> student -> examiner -> not sure
        if action == 'select_word':
            self._task_ch_pho, self._task_spelling = self._chance_player.select_word
            self._player_action = _PLAYER_ACTION('tutor', 'decide_difficulty_level')
        elif action == 'decide_difficulty_level':
            self._student_feedback = {}  # 每次转换难度要把feedback清空
            self._current_difficulty_setting = self._tutor_player.decide_difficulty_level(self._current_game_round)
            self._total_attempt = self._current_difficulty_setting['attempts']
            self._player_action = _PLAYER_ACTION('student', 'student_spelling')
        elif action == 'student_spelling':
            if self._difficulty_change:  # 在转换难度的时候才重新初始化
                if self._current_game_round > 1:  # 难度变化的时候要先让学生记忆，否则masks就清空了
                    print('--------------------------start training-----------------------------------')
                    thread = threading.Thread(target=self._student_player.student_memorizing())
                    thread.start()
                    event.wait()  # 等待事件被触发
                    print('---------------------training finished, continue playing-----------------------------------')
                    thread.join()  # 等待子线程结束
                    # reinitialize student player for getting the new difficulty setting
                self._student_player = StudentPlayer(player_id=2, player_name='student',
                                                     chinese_phonetic=self._task_ch_pho,
                                                     target_english=self._task_spelling,
                                                     current_difficulty_setting=self._current_difficulty_setting)  # reinitialize student player
                if self._current_game_round == 2:  # 只在难度二的时候确定迷惑字母
                    self._available_letter = self._student_player.letter_space  # 更新难度
                self._difficulty_change = False
            # print('学生此时得到的反馈是', self._student_feedback)
            self._student_spelling = self._student_player.student_spelling(self._student_feedback)
            self._player_action = _PLAYER_ACTION('examiner', 'give_feedback')
        elif action == 'give_feedback':
            self._student_feedback, self._tutor_feedback = self._examiner_player.give_feedback(self._student_spelling,
                                                                                               self._task_spelling)

            # 转换玩家的条件 受到回答的准确度，机会次数，还有游戏的round决定下一个玩家应该是什么
            if self._tutor_feedback[0] != 1.0:  # spelling not correct
                if self._current_attempt < self._total_attempt:  # 机会次数还没用完
                    self._current_attempt += 1
                    self._player_action = _PLAYER_ACTION('student', 'student_spelling')
                else:
                    if self._current_game_round < 4:  # 如果机会次数已经用完了
                        self._difficulty_change = True
                        self._current_game_round += 1
                        self._current_difficulty_level += 1
                        self._current_attempt = 1
                        self._player_action = _PLAYER_ACTION('tutor', 'decide_difficulty_level')
                    else:  # 游戏结束,然后就要继续初始化了
                        self._game_over = True
                        print('game over')
            else:  # 如果拼写正确
                if self._current_game_round < 4:
                    self._difficulty_change = True
                    self._current_game_round += 1
                    self._current_difficulty_level += 1
                    self._current_attempt = 1
                    self._player_action = _PLAYER_ACTION('tutor', 'decide_difficulty_level')
                else:
                    self._game_over = True
                    print('game over')
        return self._player_action

    @property
    def current_difficulty_setting(self):
        return self._current_difficulty_setting

    @property
    def stu_spelling(self) -> str:
        """Returns True if the game is over."""
        return self._student_spelling

    @property
    def stu_feedback(self) -> Union[None, Dict[str, int]]:
        """Returns True if the game is over."""
        return self._student_feedback

    @property
    def tutor_feedback(self) -> List[float]:
        """Returns True if the game is over."""
        return self._tutor_feedback
