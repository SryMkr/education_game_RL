"""Tests for open_spiel.python.algorithms.tabular_qlearner."""

from absl.testing import absltest  # 测试工具
import numpy as np
from open_spiel.python import rl_environment  # 只要是加入agent都要使用这个包，所以最好搞明白这个文件里有什么
from open_spiel.python.algorithms import tabular_qlearner  # 从算法中导入这个qlearner的包
import pyspiel

# Fixed seed to make test non stochastic.
SEED = 10000

# A simple two-action game encoded as an EFG game. Going left gets -1, going right gets a +1.
SIMPLE_EFG_DATA = """
  EFG 2 R "Simple single-agent problem" { "Player 1" } ""
  p "ROOT" 1 1 "ROOT" { "L" "R" } 0
    t "L" 1 "Outcome L" { -1.0 }
    t "R" 2 "Outcome R" { 1.0 }
"""


class QlearnerTest(absltest.TestCase):

    def test_simple_game(self):
        game = pyspiel.load_efg_game(SIMPLE_EFG_DATA)  # 加载游戏
        env = rl_environment.Environment(game=game)  # 都要传入一个游戏环境

        agent = tabular_qlearner.QLearner(0, game.num_distinct_actions())
        total_reward = 0

        for _ in range(100):  # 进行100轮游戏
            total_eval_reward = 0
            for _ in range(1000):
                time_step = env.reset()  # 初始化游戏
                while not time_step.last():  # 如果游戏还没结束
                    agent_output = agent.step(time_step)   # 接受这一步
                    time_step = env.step([agent_output.action])  # 环境中的一步
                    total_reward += time_step.rewards[0]  # 获得一个奖励
                agent.step(time_step)
            self.assertGreaterEqual(total_reward, 75)
            for _ in range(1000):
                time_step = env.reset()
                while not time_step.last():
                    agent_output = agent.step(time_step, is_evaluation=True)
                    time_step = env.step([agent_output.action])
                    total_eval_reward += time_step.rewards[0]
            self.assertGreaterEqual(total_eval_reward, 250)


if __name__ == "__main__":
    np.random.seed(SEED)
    absltest.main()
