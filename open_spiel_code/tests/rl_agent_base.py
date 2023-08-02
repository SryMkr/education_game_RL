"""
参考：https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/rl_agent.py
该方法是所有创建agent的基类包，自己创建任何agent都要实现以下的功能，为了进一步理解agent需要分析一个random agent的实例
Reinforcement Learning (RL) Agent Base for Open Spiel.
"""

import abc  # 抽象类的包，只能继承，不需要实现方法
import collections  # 特殊的容器类型

# 定义一个具名元组，（元组名字，[元组参数1，2]）类似于 step_output=("action", "probs") 输出的是动作，动作的概率分布
StepOutput = collections.namedtuple("step_output", ["action", "probs"])


class AbstractAgent(metaclass=abc.ABCMeta):
    """ Abstract base class for Open Spiel RL agents."""

    @abc.abstractmethod  # 声明该方法是一个抽象类，也就是说定义一个agent必须输入playerID，其他的都是可选择的，一般也需要输入可选择的动作
    def __init__(self,
                 player_id,
                 session=None,
                 observation_spec=None,
                 name="agent",
                 **agent_specific_kwargs):
        """ Initializes agent.
        Args:
          player_id: integer, mandatory. Corresponds to the player's position in the game and is used to index the observation list. 使用整数索引可以直接使用该索引得到某一个agent的observation
          session: optional Tensorflow session. 不适用
          observation_spec: optional dict containing observation specifications. 保存当前玩家的observation
          name: string. Must be used to scope TF variables. Defaults to `agent`. agent的命名，在TF中需要索引
          **agent_specific_kwargs: optional extra args. 其他需要加的参数
    """

    # 直接接受一个TimeStep的信息
    @abc.abstractmethod
    def step(self, time_step, is_evaluation=False):
        """ Returns action probabilities and chosen action at `time_step`. 返回动作的概率，并选择一个动作
        
        Agents should handle `time_step` and extract the required part of the`time_step.observations` field. This flexibility enables algorithms which rely on opponent observations / information, e.g. CFR.
        agent从TimeStep中提取observation成员的信息
        
        `is_evaluation` can be used so agents change their behaviour for evaluation purposes, e.g.: preventing exploration rate decaying during test and insertion of data to replay buffers.
        这个参数的意义是：因为在训练阶段，包括exploring和exploiting，在测试阶段就不需要exploiting，直接使用exploring就可以
        Arguments:
          time_step: an instance of rl_environment.TimeStep. 接受environment给的TimeStep结构体
          
          is_evaluation: bool indicating whether the step is an evaluation routine,  as opposed to a normal training step.设定是训练阶段还是评估阶段
        
        Returns: 返回值
          A `StepOutput` for the current `time_step`. 返回的是一个具名元组
        """
