import numpy as np
from citysim3d.envs import Env
from citysim3d.spaces import BoxSpace, TranslationAxisAngleSpace


class NormalizedEnv(Env):
    def __init__(self, env):
        self.__dict__.update(env.__dict__)
        self.env = env

    def step(self, action):
        if isinstance(self.env.action_space, (BoxSpace, TranslationAxisAngleSpace)):
            lb, ub = self.env.action_space.low, self.env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            raise NotImplementedError
        return self.env.step(scaled_action)

    def reset(self, state=None):
        return self.env.reset(state=state)

    def get_state(self):
        return self.env.get_state()

    def set_state(self, state):
        self.env.set_state(state)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        if isinstance(self.env.action_space, BoxSpace):
            ub = np.ones(self.env.action_space.shape)
            action_space = BoxSpace(-1 * ub, ub)
        elif isinstance(self.env.action_space, TranslationAxisAngleSpace):
            ub = np.ones(self.env.action_space.shape)
            action_space = TranslationAxisAngleSpace(-1 * ub, ub,
                                                     axis=self.env.action_space.axis,
                                                     dtype=self.env.action_space.dtype)
        else:
            raise NotImplementedError
        return action_space

    @property
    def observation_space(self):
        return self.env.observation_space
