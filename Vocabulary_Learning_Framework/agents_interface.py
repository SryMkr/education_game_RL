"""
1: agent base abstract class, and per agent interface
2: initial some parameters that will be regularly used in agents instance
# In a nutshell: An agent normally has
    (1) attributes: player_ID, player_Name, **agent_specific_kwargs, all implemented in __init__ function
    (2) step function: A: parameter: get the observation of environment time step
                       B: a policy get the observation and provide the action probabilities, then agent select an action based on probabilities
                       (observation->action probabilities->action)
    compared with the (uniform_random) agent that have different policy


思考将agent，可以分为直接初始化，然后结合策略，选择动作的道路
定义一下这几类agents
"""

import abc
from typing import List, Tuple, Dict
import random


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
             A `StepOutput` for the current `time_step`. (an action or information) !!!!!!!!!!!
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
    """
    是为了组合学习新单词和组合旧单词的功能
    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str
                 ):
        super().__init__(player_id, player_name)
        """Initializes TaskCollector agent."""

    @abc.abstractmethod
    def step(self, time_step) -> int:
        """
                    :return: the action index
                    """
        pass


class PresentWordInterface(AgentAbstractBaseClass):
    """ select one word from session data, there are four method (1) sequential (2) random (3) easy to hard (4)DDA
    legal actions: the word length in one session,
    Observation：TimeStep [session data，accuracy, feedback]
    Policy：random, sequential, easy to hard, dynamic difficulty adjustment
    Output：action: the selected word length
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
        self._action: int = 0

    @abc.abstractmethod
    def action_policy(self, time_step):
        """ StepOutput = collections.namedtuple("step_output", ["action", "probabilities"]) """

    @abc.abstractmethod
    def step(self, time_step) -> int:
        """
        if correctly answer at hardest test level, the agent need to remove the task from session

        :returns the action: the length of task, integer"""
        pass


class StudentInterface(AgentAbstractBaseClass):
    """ optional information:  available letter,  ['蜘蛛 n s p aɪ d ɝ', 's p i d e r']

    今天先实现一个随机的拼写，然后把与环境的交互定义好，完成一次整体的流程
    legal actions: the index of [a,b,c,d,e.................x,y,z]
    Observation：time step [conditions(chinese,phonetic,POS)，answer_length, available_letter(optional), accuracy, letter_mark]
    policy：random, perfect, forgetting
    output：actions: the index of [a,b,c,d,e.................x,y,z], for example, [2,3,4,5,6,2,1]
    State: provide spelling
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
             self._actions: the index of alphabet
             self._policy, student type: random, forget, perfect
        """
        self._policy: str = policy

    @abc.abstractmethod
    def stu_spell(self, time_step) -> List[int]:
        """
        :return: student spelling
        """

    @abc.abstractmethod
    def stu_learn(self, time_step) -> None:
        """
        update n-grams based on feedback
        """

    @abc.abstractmethod
    def step(self, time_step) -> List[int]:
        """:returns actions"""


class ExaminerInterface(AgentAbstractBaseClass):
    """Examiner Interface
        Legal actions: [0，1], where 0 presents wrong, 1 denotes right
        Observation：TimeStep [student_spelling，answer]
        Policy：no policy
        Output：actions: for example, [0,1,1,0,1,1,1]
        State: calculate the accuracy and completeness
    """
    def __init__(self,
                 player_id: int,
                 player_name: str):
        super().__init__(player_id,
                         player_name)
        """Initializes examiner agent"""

    @abc.abstractmethod
    def step(self, time_step) -> List[int]:
        """
        :returns the actions list, consisting of 0 and 1
        """
        pass
