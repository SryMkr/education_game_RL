"""
1: agent base abstract class, and per agent interface
2: initial some parameters that will be regularly used in agents instance
# In a nutshell: An agent normally has
    (1) attributes: player_ID, player_Name, **agent_specific_kwargs, all implemented in __init__ function
    (2) step function: A: parameter: get the observation of environment time step
                       B: a policy get the observation and provide the action probabilities, then agent select an action based on probabilities
                       (observation->action probabilities->action)
    compared with the (uniform_random) agent that have different policy

"""

import abc
from typing import List, Tuple, Dict


class AgentAbstractBaseClass(metaclass=abc.ABCMeta):
    """Abstract base class for all agents."""

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 **agent_specific_kwargs):
        """Initializes agent.

                Args:
                    player_id: zero-based integer，for index agent
                    player_name: string.
                    **agent_specific_kwargs: optional extra args.
                """
        self._player_id: int = player_id
        self._player_name: str = player_name

    @abc.abstractmethod
    def step(self, time_step):
        """
           Agents should handle `time_step` and extract the required part of the
           `time_step.observations` field.

           Arguments:
             time_step: an instance of rl_environment.TimeStep.
           Returns:
             A `StepOutput` for the current `time_step`. (action or actions) !!!!!!!!!!!
           """

    @property
    def player_id(self) -> int:
        """
        :return: the player_id
        """
        return self._player_id

    @property
    def player_name(self) -> str:
        """
        :return: the player_name
        """
        return self._player_name


class SessionCollectorInterface(AgentAbstractBaseClass):
    """session player
    legal actions: the number of sessions,
    Observation：legal actions： List[int]
    Policy：random, sequential
    Output：action: the session index
    State: read the session data based on the session index
    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 policy: str
                 ):
        super().__init__(player_id, player_name)
        """Initializes TaskCollector agent."""
        self._policy = policy

    @abc.abstractmethod
    def step(self, time_step) -> int:
        """
                    :return: the session index
                    """
        pass


class PresentWordInterface(AgentAbstractBaseClass):
    """ select one word from session data, there are four method (1) sequential (2) random (3) easy to hard (4)DDA
    legal actions: the word length in one session,
    Observation：TimeStep [session data，examiner feedback, last word difficulty]
    Policy：random, sequential, easy to hard, dynamic difficulty adjustment
    Output：action: the selected word length (difficulty level)
    State: select a task from session data
    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 policy: str,
                 ):
        super().__init__(player_id, player_name)
        '''
        :args: 
        self._policy: str, the type of policy
        '''
        self._policy = policy

    @abc.abstractmethod
    def define_difficulty(self, time_step) -> Dict[int, str]:
        """tutor agent define the difficulty of each task
        difficulty definition: the length of word
        """
        pass

    @abc.abstractmethod
    def action_policy(self, time_step) -> str:
        """ the length of task """

    @abc.abstractmethod
    def step(self, time_step) -> List[str]:
        """
        if correctly answer at hardest test level, the agent need to remove the task from session

        :returns the action: the task, str"""
        pass


class StudentInterface(AgentAbstractBaseClass):
    """ optional information:  [available letter],, length of answer,  ['s p aɪ d ɝ']

    legal actions: the index of [a,b,c,d,e.................x,y,z]
    Observation：time step [[phonetic，answer_length, available_letter], [accuracy, letter_mark]]
    Policy：random, excellent, forgetting
    Output：actions: the index of [a,b,c,d,e.................x,y,z], for example, [2,3,4,5,6,2,1]
    State: convert index to letter
    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 policy: str,
                 ):
        super().__init__(player_id, player_name)
        """Initializes student agent.

        Args:
             self._policy, student type: random, forget, perfect
        """
        self._policy: str = policy
        self.position_condition: List[str] = []

    @abc.abstractmethod
    def stu_spell(self, time_step) -> List[int]:
        """
        :return: student spelling
        """

    @abc.abstractmethod
    def stu_learn(self, time_step) -> None:
        """
        update table based on feedback!!!!!!!!!!!!!!! need to be finished
        """

    @abc.abstractmethod
    def step(self, time_step) -> List[int]:
        """:returns actions!!!!!!!!!!!!!!!!!!!"""


class ExaminerInterface(AgentAbstractBaseClass):
    """Examiner Interface
        Legal actions: [0，1], where 0 presents wrong, 1 denotes right
        Observation：TimeStep [student_spelling，answer]
        Policy：no policy
        Output：actions: for example, [0,1,1,0,1,1,1]!!!!!!!!!!!!!!!!
        State: calculate the accuracy and completeness
    """

    def __init__(self,
                 player_id: int,
                 player_name: str):
        super().__init__(player_id,
                         player_name)
        """Initializes examiner agent"""

    @abc.abstractmethod
    def step(self, time_step) -> Tuple[List[int], float, float]:
        """
        :returns the actions list, consisting of 0 and 1
        """
        pass
