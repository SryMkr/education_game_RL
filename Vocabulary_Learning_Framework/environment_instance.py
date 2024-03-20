"""
environment instance consists of two fundamental functions: step，get_time_step.

step：（1） apply the action receiving from agent to change the current state of environment
get_time_step:（2）read information from the new state as the agents information

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
    """ create the interactive environment"""

    def __init__(self,
                 vocabulary_book_path,
                 vocabulary_book_name,
                 chinese_setting,
                 phonetic_setting,
                 POS_setting,
                 english_setting,
                 history_words_number,
                 review_words_number,
                 sessions_number,
                 ):
        super().__init__(vocabulary_book_path,
                         vocabulary_book_name,
                         chinese_setting,
                         phonetic_setting,
                         POS_setting,
                         english_setting,
                         history_words_number,
                         review_words_number,
                         sessions_number,
                         )

    def new_initial_state(self):
        """ calling the state, and pass the history words"""
        return VocabSpellState(self.history_words, self.review_words_number, self.sessions_number)

    def reset(self):
        """ initialize the state of environment"""
        self._state = self.new_initial_state()  # get the initial state of environment
        self._should_reset = False
        # initialize the observations, and read from state object
        observations = {"history_words": None, "current_session_num": None,
                        "sessions_number": self._state.sessions_number,
                        "review_words_number": self._state.review_words_num,
                        "current_session_words": None, "legal_actions": [],
                        "current_player": self._state.current_player,
                        "condition": None, "answer": None,
                        "answer_length": None, "student_spelling": None,
                        "examiner_feedback": None, "history_information": None}

        # add the legal action of each player
        for player_ID in range(self._player_num):
            observations["legal_actions"].append(self._state.legal_actions(player_ID))

        return TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=StepType.FIRST)

    def get_time_step(self):
        observations = {"history_words": self._state.history_words,
                        "sessions_number": self._state.sessions_number,
                        "review_words_number": self._state.review_words_num,
                        "current_session_num": self._state.current_session_num,
                        "current_session_words": self._state.current_session_words, "legal_actions": [],
                        "current_player": self._state.current_player,
                        "condition": self._state.condition,
                        "answer": self._state.answer,
                        "answer_length": self._state.answer_length, "student_spelling": self._state.stu_spelling,
                        "examiner_feedback": self._state.examiner_feedback,
                        "history_information": self._state.history_information
                        }

        # add the legal action of each player
        for player_ID in range(self._player_num):
            observations["legal_actions"].append(self._state.legal_actions(player_ID))

        rewards = self._state.rewards  # how to define the rewards?!!!!!!!!!
        discounts = self._discount
        step_type = StepType.LAST if self._state.is_terminal else StepType.MID  # indicate the stage of environment
        self._should_reset = step_type == StepType.LAST  # True, if game terminate

        if step_type == StepType.LAST:
            # what to do if the game terminate !!!!!!!!!!!!!!!!!
            pass

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
