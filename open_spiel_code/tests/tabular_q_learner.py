"""
1：两个玩家，一人一个Q表，训练好模型
2：让训练好的模型和一个随机策略比较
3：RL is based on python/examples/independent_tabular_qlearning.py
"""

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner
import numpy as np

# Create the environment
env = rl_environment.Environment(game="tic_tac_toe")  # 返回值就是一个环境对象
num_players = env.num_players  # 获得游戏的玩家数量
num_actions = env.action_spec()["num_actions"]  # 返回的是个字典，要用key来获得自己想要的属性

# Create the agents
agents = [
    # 在Q表中，就只有一个类，就是底下这个agent的类，创建两个，因为有两个agent
    tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
    for idx in range(num_players)
]

# Train the Q-learning agents in self-play.
for cur_episode in range(25000):  # 训练n次
    if cur_episode % 1000 == 0:  # 每1000次，输出以下的语句
        print(f"Episodes: {cur_episode}")
    time_step = env.reset()  # 将环境reset为初始环境
    while not time_step.last():  # 只要不是最后一步
        player_id = time_step.observations["current_player"]  # 也是个字典，能获得当前的玩家
        agent_output = agents[player_id].step(time_step)  # 输入的是state的时间步
        time_step = env.step([agent_output.action])  # 环境也要向前走一步
    # Episode is over, step all agents with final info state.
    for agent in agents:
        agent.step(time_step)
print("Done!")

# Evaluate the Q-learning agent VS a random agent.
from open_spiel.python.algorithms import random_agent  # 这个random agent需要了解，随时用的着

eval_agents = [agents[0], random_agent.RandomAgent(1, num_actions, "Entropy Master 2000")]
time_step = env.reset()
total_eval_reward = [0, 0]  # 用来记录玩家的总分
for _ in range(10000):  # 进行n轮游戏
    while not time_step.last():
        # print("")
        # print(env.get_state)
        player_id = time_step.observations["current_player"]
        # Note the evaluation flag. A Q-learner will set epsilon=0 here.
        agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)  # 获得agent的输出
        # print(f"Agent {player_id} chooses {env.get_state.action_to_string(agent_output.action)}")
        time_step = env.step([agent_output.action])  # 将动作运用到环境中
    total_eval_reward = np.sum([total_eval_reward, time_step.rewards], axis=0).tolist()  # 获得一个奖励
    # print("")
    # print(env.get_state)
print(total_eval_reward)

