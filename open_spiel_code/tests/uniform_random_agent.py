"""A bot that chooses uniformly at random from legal actions."""

import pyspiel


class UniformRandomBot(pyspiel.Bot):
  """Chooses uniformly at random from the available legal actions."""

  def __init__(self, player_id, rng):
    """Initializes a uniform-random bot.

    Args:
      player_id: The integer id of the player for this bot, e.g. `0` if acting as the first player.
      rng: A random number generator supporting a `choice` method, e.g.
        `np.random`
    """
    pyspiel.Bot.__init__(self)
    self._player_id = player_id
    self._rng = rng

  def restart_at(self, state):
    pass

  def player_id(self):
    return self._player_id

  def provides_policy(self):
    return True

  def step_with_policy(self, state):
    """Returns the stochastic policy and selected action in the given state.

    Args:
      state: The current state of the game.

    Returns:
      A `(policy, action)` pair, where policy is a `list` of `(action, probability)` pairs for each legal action, with `probability = 1/num_actions`
      The `action` is selected uniformly at random from the legal actions, or `pyspiel.INVALID_ACTION` if there are no legal actions available.
    """
    legal_actions = state.legal_actions(self._player_id)
    if not legal_actions:
      return [], pyspiel.INVALID_ACTION
    p = 1 / len(legal_actions)
    policy = [(action, p) for action in legal_actions]  # 返回的policy是 [(action,p),(action,p),(action,p),(action,p)]
    action = self._rng.choice(legal_actions) # 在这里已经选择了动作了
    return policy, action  # 所以返回的是每一个动作索引及其概率的元组，后面一个是它选择的动作。

  def step(self, state):
    return self.step_with_policy(state)[1]  # 代表的是选择出来的动作
