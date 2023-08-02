"""A vectorized RL Environment. 矢量化的强化学习环境是指在强化学习中，同时处理多个状态或多个智能体的环境。传统的强化学习环境一次只能处理一个状态或一个智能体的动作，这样在训练过程中会逐步进行，效率较低。

而矢量化的环境能够同时处理多个状态或多个智能体的动作，通常通过向量化运算来实现高效的数据处理。这样做的好处是在单次操作中处理多个状态或动作，大大提高了计算效率，加快了训练过程。

矢量化的强化学习环境可以应用于诸如多智能体博弈、多任务学习和分布式训练等场景，通过并行处理多个状态或智能体的信息，加速了模型的训练，提高了训练效率。同时，矢量化的环境还有助于利用硬件加速（如GPU或TPU）
来进一步提高训练速度，从而更好地应用于复杂的强化学习任务。

---------------------------加速训练不是我现在考虑的问题，但是一个环境中有step,reset方法，有获取状态参数的方法。这个文件目前不是很重要----------------------------------------------------------

"""

class SyncVectorEnv(object):
  """A vectorized RL Environment.

  This environment is synchronized - games do not execute in parallel. Speedups
  are realized by calling models on many game states simultaneously.
  """

  def __init__(self, envs):
    if not isinstance(envs, list):
      raise ValueError(
          "Need to call this with a list of rl_environment.Environment objects")
    self.envs = envs

  def __len__(self):
    return len(self.envs)
    
  def observation_spec(self):
  return self.envs[0].observation_spec()

  @property
  def num_players(self):
    return self.envs[0].num_players

  def step(self, step_outputs, reset_if_done=False):
    """Apply one step.

    Args:
      step_outputs: the step outputs
      reset_if_done: if True, automatically reset the environment
          when the epsiode ends

    Returns:
      time_steps: the time steps,
      reward: the reward
      done: done flag
      unreset_time_steps: unreset time steps
    """
    time_steps = [
        self.envs[i].step([step_outputs[i].action])
        for i in range(len(self.envs))
    ]
    reward = [step.rewards for step in time_steps]
    done = [step.last() for step in time_steps]
    unreset_time_steps = time_steps  # Copy these because you may want to look
                                     # at the unreset versions to extract
                                     # information from them

    if reset_if_done:
      time_steps = self.reset(envs_to_reset=done)

    return time_steps, reward, done, unreset_time_steps

  def reset(self, envs_to_reset=None):
    if envs_to_reset is None:
      envs_to_reset = [True for _ in range(len(self.envs))]

    time_steps = [
        self.envs[i].reset()
        if envs_to_reset[i] else self.envs[i].get_time_step()
        for i in range(len(self.envs))
    ]
    return time_steps
