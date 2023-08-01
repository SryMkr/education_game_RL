""" An observation of a game. This is intended to be the main way to get observations of states in Python. observation是TimeStep中的一个成员  
这个文件其实没有什么有价值的信息，其实就是创建一个方法构造observation中应该有的三个成员【information，legal action, current_player】
The usage pattern is as follows:  
0. Create the game we will be playing。 首先要有一个游戏  
1. Create each kind of observation required, using `make_observation`。  使用make_observation方法构造observation    
2. Every time a new observation is required, call: `observation.set_from(state, player)` 该方法是为了根据当前的state，得到某个玩家的observation
   The tensor contained in the Observation class will be updated with an
   observation of the supplied state. This tensor is updated in-place, so if
   you wish to retain it, you must make a copy.
The following options are available when creating an Observation:
 - perfect_recall: if true, each observation must allow the observing player to reconstruct their history of actions and observations. 能看到历史所有的记录
 - public_info: if true, the observation should include public information。所有agent都可以看到的信息
 - private_info: specifies for which players private information should be。 某个agent的特定信息
   included - all players, the observing player, or no players
 - params: game-specific parameters for observations。还有一些其他的游戏参数可以看到
We ultimately aim to have all games support all combinations of these arguments.
However, initially many games will only support the combinations corresponding
to ObservationTensor and InformationStateTensor:
 - ObservationTensor: perfect_recall=False, public_info=True,
   private_info=SinglePlayer
 - InformationStateTensor: perfect_recall=True, public_info=True,
   private_info=SinglePlayer
Three formats of observation are supported:
a. 1-D numpy array, accessed by `observation.tensor`
b. Dict of numpy arrays, accessed by `observation.dict`. These are pieces of the
   1-D array, reshaped. The np.array objects refer to the same memory as the
   1-D array (no copying!).
c. String, hopefully human-readable (primarily for debugging purposes)
For usage examples, see `observation_test.py`.
"""

import numpy as np
import pyspiel

# Corresponds to the old information_state_XXX methods.
# 根据设定的perfect_recall的真假，来决定他是一个observation还是一个information
# 区别在于能不能回溯过去的信息步骤
"""
首先游戏可以分为两类，一种是perfect information game是所有的信息大家都是知道的例如棋盘，第二种是imperfect 
information game 是有私人的信息，但是整个state是可以观察到的。这个板块只是提供了一下必要的借口，第三种是
玩家自己有各自的私人信息，但是整个state是partial的
对于现在的我的游戏只要考虑一件事即可：根据有没有私人信息，来区分完美信息游戏和不完美信息游戏
而且即使是不完美的信息，param参数根本不需要，因为可以直接加入到public information 里面，所以对于不完美信息游戏来说
其实就是公共信息，私人信息，以及过去的步骤信息能不能回溯
根本上不要调用open-spiel已经分好的游戏类别，完全没有必要，设定observation只需要做以下几件事
（1）定一个一个游戏的observer类
（2）在这个类中实现 self.tensor, self.dic, 属性
（3）在这个类中实现 set_from， string_from 方法 第一个是为了让游戏看，第二个是为了让human看 就妥了
"""

INFO_STATE_OBS_TYPE = pyspiel.IIGObservationType(perfect_recall=True)


# 这里可能是为了直接调用已经写好的observation类
class _Observation:
    """ Contains an observation from a game."""
    # 定义三个参数，环境，iig的游戏观察类型， 环境参数
    def __init__(self, game, imperfect_information_observation_type, params):
        if imperfect_information_observation_type is not None:
            obs = game.make_observer(imperfect_information_observation_type, params)
        else:
            obs = game.make_observer(params)
        self._observation = pyspiel._Observation(game, obs)
        self.dict = {}
        if self._observation.has_tensor():
            self.tensor = np.frombuffer(self._observation, np.float32)
            offset = 0
            for tensor_info in self._observation.tensors_info():
                size = np.product(tensor_info.shape, dtype=np.int64)
                values = self.tensor[offset:offset + size].reshape(tensor_info.shape)
                self.dict[tensor_info.name] = values
                offset += size
        else:
            self.tensor = None

    def set_from(self, state, player):
        self._observation.set_from(state, player)

    def string_from(self, state, player):
        return (self._observation.string_from(state, player)
                if self._observation.has_string() else None)

    def compress(self):
        return self._observation.compress()

    def decompress(self, compressed_observation):
        self._observation.decompress(compressed_observation)


# 这是为了使用已经定义好的游戏类别，直接可以返回一个observatone
def make_observation(game, imperfect_information_observation_type=None, params=None):
    if hasattr(game, 'make_py_observer'):  # 如果对象有该属性返回 True
        return game.make_py_observer(imperfect_information_observation_type, params)
    else:  # 否则返回 False
        return _Observation(game, imperfect_information_observation_type, params or {})


# 这里的类建议的是如何实现在imperfect information observations中，所有玩家都知道的公共信息
class IIGObserverForPublicInfoGame:
    """Observer for imperfect information observations of public-info games."""
    # 第一个参数是观察类型，
    def __init__(self, iig_obs_type, params):
        if params:  # 根本不需要params 是直接定义在了public information中的
            raise ValueError(f'Observation parameters not supported; passed {params}')
        self._iig_obs_type = iig_obs_type  # 根据open-spiel的定义看是哪一类
        self.tensor = None  # tensor信息类
        self.dict = {}  # 字典信息类

    def set_from(self, state, player):  # 自己实现如何让用tensor展示observation
        pass

    def string_from(self, state, player):  # 自己实现如何用字符串展示observation
        del player
        if self._iig_obs_type.public_info:
            return state.history_str()
        else:
            return ''  # No private information to return
