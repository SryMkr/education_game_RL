"""
game environment instance
"""
from environment_interface import EnvironmentInterface
from state_instance import VocabSpellState
import collections
import enum


class StepType(enum.Enum):
    """Defines the status of a `TimeStep` within vocabulary book."""

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
    __slots__ = ()

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def current_player(self):
        return self.observations["current_player"]


class VocabSpellGame(EnvironmentInterface):
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

    # initialize the state of game,实例化状态对象，所以在环境中可以直接调用状态的属性
    def new_initial_state(self):
        return VocabSpellState()

    def reset(self):
        self._state = self.new_initial_state()
        self._should_reset = False
        # initialize the observations, and read from state object,也要读取所有的历史信息
        observations = {"vocab_sessions": self.vocab_sessions, "current_session_num": self._state.current_session_num,
                        "legal_action": self._state.legal_action, "current_player": self._state.current_player,
                        "vocab_session": None}
        return TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=StepType.FIRST)

    def get_time_step(self):
        # 这里必须可以读取改变后的state的所有信息，return TimeStep
        # 而state的变化，和get_time_step共同封装到step，和reset中
        # 如何动态创建字典里的元素，而不需要自己添加？,如何一开始就保存好字典中需要的所有字段
        observations = {"vocab_sessions": self.vocab_sessions, "current_session_num": self._state.current_session_num,
                        "legal_action": self._state.legal_action, "current_player": self._state.current_player,
                        "vocab_session": self._state.vocab_session}

        rewards = self._state.rewards  # how to define the rewards?!!!!!!!!!
        discounts = self._discount
        step_type = StepType.LAST if self._state.is_terminal else StepType.MID
        self._should_reset = step_type == StepType.LAST  # True, if game terminate
        if step_type == StepType.LAST:
            pass
            # 还要记录包含所有的信息，那么在state中应该有个history参数
        return TimeStep(
            observations=observations,
            rewards=rewards,
            discounts=discounts,
            step_type=step_type)

    def step(self, information):
        if self._should_reset:
            return self.reset()
        self._state.apply_information(information)  # state对象复杂接收action，并引起一些其他参数的变化
        return self.get_time_step()
