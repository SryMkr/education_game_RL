"""
1：定义一个采取随机策略的agent
"""

import numpy as np
from open_spiel.python import rl_agent  # 定义agent的基类


class RandomAgent(rl_agent.AbstractAgent):  # 继承agent中的基类
    """Random agent class."""
    # 输入有很多，基本上必须输入的参数就只有playerID，以及能够采取的合法的动作
    def __init__(self, player_id, num_actions, name="random_agent"):
        assert num_actions > 0  # 必须有合法的动作可以做
        self._player_id = player_id  # 获得玩家的ID
        self._num_actions = num_actions  # 获得agent能够采取的动作

    def step(self, time_step, is_evaluation=False): # is_evaluation如果是评估的话，就不需要exploration，只是exploitation了
        # If it is the end of the episode, don't select an action.
        if time_step.last():  # 代表游戏已经结束
            return

        # Pick a random legal action.
        cur_legal_actions = time_step.observations["legal_actions"][self._player_id]
        action = np.random.choice(cur_legal_actions)  # 随机选择一个动作
        probs = np.zeros(self._num_actions)  # 设定概率数组
        probs[cur_legal_actions] = 1.0 / len(cur_legal_actions)  # 随机选择一个动作的概率是1/动作总数
        # 输出的是一个元组 （action，probability）
        return rl_agent.StepOutput(action=action, probs=probs)
