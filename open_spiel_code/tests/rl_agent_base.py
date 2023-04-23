"""
该方法是所有创建agent的基类包，自己创建任何agent都要实现以下的功能
Reinforcement Learning (RL) Agent Base for Open Spiel.
"""

import abc  # 抽象类的包，只能继承，不需要实现方法
import collections  # 特殊的容器类型

# 定义一个元组，（元组名字，[元组参数1，2]）类似于 step_output=("action", "probs") 输出的是动作，以及采取该动作的概率
StepOutput = collections.namedtuple("step_output", ["action", "probs"])


class AbstractAgent(metaclass=abc.ABCMeta):
    """Abstract base class for Open Spiel RL agents."""

    @abc.abstractmethod  # 声明该方法是一个抽象类，也就是说定义一个agent必须输入playerID，其他的都是可选择的，一般也需要输入可选择的动作
    def __init__(self,
                 player_id,
                 session=None,
                 observation_spec=None,
                 name="agent",
                 **agent_specific_kwargs):
        """Initializes agent.
    Args:
      player_id: integer, mandatory. Corresponds to the player position in the game and is used to
      index the observation list.
      session: optional Tensorflow session.
      observation_spec: optional dict containing observation specifications.
      name: string. Must be used to scope TF variables. Defaults to `agent`.
      **agent_specific_kwargs: optional extra args.
    """

    # 接受两个参数，一个是接受agent的observation，（time_step.observations）
    @abc.abstractmethod
    def step(self, time_step, is_evaluation=False):
        """Returns action probabilities and chosen action at `time_step`.
    Agents should handle `time_step` and extract the required part of the
    `time_step.observations` field. This flexibility enables algorithms which
    rely on opponent observations / information, e.g. CFR.

    `is_evaluation` can be used so agents change their behaviour for evaluation
    purposes, e.g.: preventing exploration rate decaying during test and
    insertion of data to replay buffers.
    Arguments:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool indicating whether the step is an evaluation routine,
        as opposed to a normal training step.
    Returns:
      A `StepOutput` for the current `time_step`.
    """
