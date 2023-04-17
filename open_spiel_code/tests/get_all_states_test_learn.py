"""Tests for open_spiel.python.algorithms.get_all_states."""
# 这个就是测试代码，测试通过了，里面的一些参数是测试人员设计的，没有太多的作用
from absl.testing import absltest
from open_spiel.python.algorithms import get_all_states
import pyspiel


class GetAllStatesTest(absltest.TestCase):  # 做测试好像都要继承这个类

    def test_tic_tac_toe_number_histories(self):  # 测试tic-tac-toe这个游戏
        game = pyspiel.load_game("tic_tac_toe")  # 加载这个游戏
        states = get_all_states.get_all_states(  # 调用获得所有states的函数
            game,
            depth_limit=-1,  # 分析game tree -1代表无限制，没有限制是因为这个游戏树并不大
            include_terminals=True,   # 肯定包括结束
            include_chance_states=False,  # 不需要发牌，回合制的你一步我一步
            to_string=lambda s: s.history_str())  # 直接就是用这种方法转换为string类型
        self.assertLen(states, 549946)  # 判断states的长度是否等于549946，是则通过，不是则报错
        states = get_all_states.get_all_states(
            game,
            depth_limit=-1,
            include_terminals=True,
            include_chance_states=False,
            to_string=str)
        self.assertLen(states, 5478)

    def test_simultaneous_python_game_get_all_state(self):
        game = pyspiel.load_game(
            "python_iterated_prisoners_dilemma(max_game_length=6)")
        states = get_all_states.get_all_states(
            game,
            depth_limit=-1,
            include_terminals=True,
            include_chance_states=False,
            to_string=lambda s: s.history_str())
        self.assertLen(states, 10921)
        states = get_all_states.get_all_states(
            game,
            depth_limit=-1,
            include_terminals=True,
            include_chance_states=False,
            to_string=str)
        self.assertLen(states, 5461)

    def test_simultaneous_game_get_all_state(self):
        game = game = pyspiel.load_game("goofspiel", {"num_cards": 3})
        states = get_all_states.get_all_states(
            game,
            depth_limit=-1,
            include_terminals=True,
            include_chance_states=False,
            to_string=lambda s: s.history_str())
        self.assertLen(states, 273)


if __name__ == "__main__":
    absltest.main()
