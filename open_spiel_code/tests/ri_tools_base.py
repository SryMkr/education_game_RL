""" Reinforcement Learning (RL) tools Open Spiel.该文件就是定了一个值的基类，并且通过这个值基类实现了常量和线性变量。"""

import abc  # 抽象类，只能继承

# 值基类，一种不变，一种线性变化
class ValueSchedule(metaclass=abc.ABCMeta):
    """ Abstract base class changing (decaying) values. 表示一种根据时间或其他参数变化的值的序列，通常用于调整模型中的超参数或权重等"""

    @abc.abstractmethod
    def __init__(self):
        """ Initialize the value schedule."""

    @abc.abstractmethod
    def step(self):
        """ Apply a potential change in the value. This method should be called every time the agent takes a training step.
    Returns:
      the value after the step.
    """

    @property
    @abc.abstractmethod
    def value(self):
        """ Return the current value."""


# 这个是常量，整个游戏都不改变的值
class ConstantSchedule(ValueSchedule):
    """ A schedule that keeps the value constant."""

    def __init__(self, value):
        super(ConstantSchedule, self).__init__()
        self._value = value

    def step(self):
        return self._value

    @property
    def value(self):
        return self._value


# 所以这个线性时间表是用来生成变量的
class LinearSchedule(ValueSchedule):
    """ A simple linear schedule."""

    def __init__(self, init_val, final_val, num_steps):
        """ A simple linear schedule.
            Once the the number of steps is reached, value is always equal to the final
            value.
    Arguments:
      init_val: the initial value.
      final_val: the final_value
      num_steps: the number of steps to get from the initial to final value.
    """
        super(LinearSchedule, self).__init__()
        self._value = init_val  # 一个数的初始值
        self._final_value = final_val   # 一个数的终止值
        assert isinstance(num_steps, int)  # 判断这个值是不是int类型
        self._num_steps = num_steps  # 获得一个时间步长
        self._steps_taken = 0  # 记录已经走了多少步
        self._increment = (final_val - init_val) / num_steps  # 每一步的增加值是多少

    # 走一步需要更新的变量
    def step(self):
        self._steps_taken += 1  # 记录走了多少步
        if self._steps_taken < self._num_steps:   # 如果还没有到结尾
            self._value += self._increment  # 每走一步加一个增量
        elif self._steps_taken == self._num_steps:  # 如果已经走到了结尾
            self._value = self._final_value  # 那直接将重点值给最后一步就成
        # 返回当前步的值
        return self._value

    #  返回当前步的值
    @property
    def value(self):
        return self._value
