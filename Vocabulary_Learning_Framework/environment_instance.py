"""
environment instance
基本上就是有两个方法： step，reset
上述每一种方法分为两步：（1） 将action应用到state中使得state发生变化，
                    （2）需要使用从新的state中构造新的TimeStep， step使用get_time_step, reset使用new_initial_state
目的都是为了构造一个TimeStep,里面包含环境中的所有信息，然后让相应的agent做选择
"""
from environment_interface import EnvironmentInterface
from state_instance import VocabSpellState
import collections
import enum


class StepType(enum.Enum):
    """Defines the status of a `TimeStep`, the state of game"""

    FIRST = 0  # Denotes the initial `TimeStep`, learn start.
    MID = 1  # Denotes any `TimeStep` that is not FIRST or LAST.
    LAST = 2  # Denotes the last `TimeStep`learn end.

    def first(self):
        return self is StepType.FIRST

    def mid(self):
        return self is StepType.MID

    def last(self):
        return self is StepType.LAST


class TimeStep(collections.namedtuple("TimeStep", ["observations", "rewards", "discounts", "step_type"])):
    """Returned with every call to `step` and `reset`.

  """
    __slots__ = ()  # constrict the attributes of class

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST


class VocabSpellGame(EnvironmentInterface):
    """ create the interactive env"""
    def __init__(self,
                 vocabulary_book_path,
                 vocabulary_book_name,
                 chinese_setting,
                 phonetic_setting,
                 POS_setting,
                 english_setting,
                 new_words_number,
                 ):
        super().__init__(vocabulary_book_path,
                         vocabulary_book_name,
                         chinese_setting,
                         phonetic_setting,
                         POS_setting,
                         english_setting,
                         new_words_number)

    # initialize the state of game, to read the state attributes
    def new_initial_state(self):
        """ calling the state, and pass the vocabulary sessions"""
        return VocabSpellState(self._vocabulary_sessions)

    def reset(self):
        self._state = self.new_initial_state()
        self._should_reset = False
        # initialize the observations, and read from state object
        observations = {"vocab_sessions": None,
                        "current_session_num": None,
                        "vocab_session": None, "legal_actions": [],
                        "current_player": self._state.current_player,
                        "condition": None, "answer": None,
                        "answer_length": None, "student_spelling": None,
                        "examiner_feedback": None, "history": None}

        for player_ID in range(self._player_num):
            # 动作是另外添加的
            observations["legal_actions"].append(self._state.legal_actions(player_ID))

        return TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=StepType.FIRST)

    def get_time_step(self):
        observations = {"vocab_sessions": self._state.vocab_sessions,
                        "current_session_num": self._state.current_session_num,
                        "current_player": self._state.current_player, "legal_actions": [],
                        "vocab_session": self._state.vocab_session, "condition": self._state.condition,
                        "answer": self._state.answer,
                        "answer_length": self._state.answer_length, "student_spelling": self._state.stu_spelling,
                        "examiner_feedback": self._state.examiner_feedback, "history": self._state.history
                        }

        for player_ID in range(self._player_num):
            """每个agent的动作不同，所以需要单独添加"""
            observations["legal_actions"].append(self._state.legal_actions(player_ID))

        rewards = self._state.rewards  # how to define the rewards?!!!!!!!!!
        discounts = self._discount
        step_type = StepType.LAST if self._state.is_terminal else StepType.MID  # 指示当前游戏状态
        self._should_reset = step_type == StepType.LAST  # True, if game terminate
        if step_type == StepType.LAST:
            pass
            # 还要记录包含所有的信息，那么在state中应该有个history参数 state._history!!!!!!!!!!!!!!!!!
        return TimeStep(
            observations=observations,
            rewards=rewards,
            discounts=discounts,
            step_type=step_type)

    def step(self, action):
        if self._should_reset:
            return self.reset()
        self._state.apply_action(action)  # (1) apply action/actions
        # (2) construct new TimeStep
        return self.get_time_step()
