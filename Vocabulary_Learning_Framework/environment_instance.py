"""
environment instance
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

    # initialize the state of game, to read the state attributes
    def new_initial_state(self):
        return VocabSpellState(self._vocabulary_sessions)

    def reset(self):
        self._state = self.new_initial_state()
        self._should_reset = False
        # initialize the observations, and read from state object, need to be finished
        observations = {"vocab_sessions": self._state.vocab_sessions,
                        "current_session_num": self._state.current_session_num,
                        "vocab_session": None, "legal_actions": [], "current_player": self._state.current_player,
                        "condition": self._state.condition, "answer": self._state.answer,
                        "answer_length": self._state.answer_length, "student_spelling": self._state.stu_spelling,
                        "letter_feedback": self._state.letter_feedback, "accuracy": self._state.accuracy,
                        "completeness": self._state.completeness,
                        }

        for player_ID in range(self._player_num):
            observations["legal_actions"].append(self._state.legal_actions(player_ID))

        return TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=StepType.FIRST)

    def get_time_step(self):
        # 这里必须可以读取改变后的state的所有信息，return TimeStep
        # 而state的变化，和get_time_step共同封装到step，和reset中

        observations = {"vocab_sessions": self._state.vocab_sessions,
                        "current_session_num": self._state.current_session_num,
                        "current_player": self._state.current_player, "legal_actions": [],
                        "vocab_session": self._state.vocab_session, "condition": self._state.condition,
                        "answer": self._state.answer,
                        "answer_length": self._state.answer_length, "student_spelling": self._state.stu_spelling,
                        "letter_feedback": self._state.letter_feedback, "accuracy": self._state.accuracy,
                        "completeness": self._state.completeness,
                        }

        for player_ID in range(self._player_num):
            observations["legal_actions"].append(self._state.legal_actions(player_ID))

        rewards = self._state.rewards  # how to define the rewards?!!!!!!!!!
        discounts = self._discount
        step_type = StepType.LAST if self._state.is_terminal else StepType.MID
        self._should_reset = step_type == StepType.LAST  # True, if game terminate
        if step_type == StepType.LAST:
            pass
            # 还要记录包含所有的信息，那么在state中应该有个history参数 state._history
        return TimeStep(
            observations=observations,
            rewards=rewards,
            discounts=discounts,
            step_type=step_type)

    def step(self, action):
        if self._should_reset:
            return self.reset()
        self._state.apply_action(action)  # apply action
        # construct new state
        return self.get_time_step()
