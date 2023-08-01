""" Joint policy denoted by the RL agents of a game. 这个文件给出的是如何将action和probability一一对应 ，当然如果是值对应的就是值，如果是概率对应的解释概率
    所以研究policy和agent的结合看本文件就可以，其实就是为了给出一个列表这个列表中对应的是{action,prob/value}的字典，本文件实现的就是
    agent通过读取observation，还有legal_action，返回一个{action,prob/value}的字典，选择动作需要自己写代码实现，其实就是在一个概率分布中选择一个动作而已，在agent的step方法中直接读取选中的action
"""

from typing import Dict

from open_spiel.python import policy
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment


class JointRLAgentPolicy(policy.Policy):
  """Joint policy denoted by the RL agents of a game.

  Given a list of RL agents of players for a game, this class can be used derive
  the corresponding (joint) policy. In particular, the distribution over
  possible actions will be those that are returned by the step() method of
  the RL agents given the state.
  """

  def __init__(self, game, agents: Dict[int, rl_agent.AbstractAgent],
               use_observation: bool):
    """ Initializes the joint RL agent policy.

    Args:
      game: The game.
      agents: Dictionary of agents keyed by the player IDs.
      use_observation: If true then observation tensor will be used as the
        `info_state` in the step() calls; otherwise, information state tensor
        will be used. See `use_observation` property of
        rl_environment.Environment.
    """
    player_ids = list(sorted(agents.keys()))
    super().__init__(game, player_ids)
    self._agents = agents
    self._obs = {
        "info_state": [None] * game.num_players(),
        "legal_actions": [None] * game.num_players()
    }
    self._use_observation = use_observation

  def action_probabilities(self, state, player_id=None):
    if state.is_simultaneous_node():
      assert player_id is not None, "Player ID should be specified."
    else:
      if player_id is None:
        player_id = state.current_player()
      else:
        assert player_id == state.current_player()

    # Make sure that player_id is an integer and not an enum as it is used to
    # index lists.
    player_id = int(player_id)

    legal_actions = state.legal_actions(player_id)

    self._obs["current_player"] = player_id
    self._obs["info_state"][player_id] = (
        state.observation_tensor(player_id)
        if self._use_observation else state.information_state_tensor(player_id))
    self._obs["legal_actions"][player_id] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    p = self._agents[player_id].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions} # 如果返回的是动作和概率的字典，那么哪里选择动作了？
    return prob_dict


class RLAgentPolicy(JointRLAgentPolicy):
  """A policy for a specific agent trained in an RL environment."""

  def __init__(self, game, agent: rl_agent.AbstractAgent, player_id: int,
               use_observation: bool):
    """Initializes the RL agent policy.

    Args:
      game: The game.
      agent: RL agent.
      player_id: ID of the player.
      use_observation: See JointRLAgentPolicy above.
    """
    self._player_id = player_id
    super().__init__(game, {player_id: agent}, use_observation)

  def action_probabilities(self, state, player_id=None):
    return super().action_probabilities(
        state, self._player_id if player_id is None else player_id)