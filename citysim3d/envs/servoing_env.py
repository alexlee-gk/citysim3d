import numpy as np
from citysim3d.envs import Env, SimpleQuadPanda3dEnv
from citysim3d.spaces import DictSpace


class ServoingEnv(Env):
    """Environment wrapper that adds additional target observation for every observation"""
    def __init__(self, env):
        self.env = env
        self._target_obs = None
        assert isinstance(self.env.observation_space, DictSpace)
        spaces = dict(self.env.observation_space.spaces)
        target_spaces = {'target_' + k: space for (k, space) in self.env.observation_space.spaces.items()}
        spaces.update(target_spaces)
        self._observation_space = DictSpace(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs.update(self._target_obs)
        return obs, reward, done, info

    def reset(self, state=None):
        obs = self.env.reset(state=state)
        self._target_obs = {'target_' + k: v for (k, v) in obs.items()}
        obs.update(self._target_obs)
        return obs

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
        return self.env.action_space

    @property
    def observation_space(self):
        return self._observation_space


class SimpleQuadPanda3dServoingEnv(ServoingEnv):
    def __init__(self, env, max_time_steps=100, distance_threshold=4.0):
        super(SimpleQuadPanda3dServoingEnv, self).__init__(env)
        assert isinstance(env, SimpleQuadPanda3dEnv)
        self.__dict__.update(env.__dict__)
        self.max_time_steps = max_time_steps
        self.distance_threshold = distance_threshold
        self._target_pos = None  # position of the target relative to the camera
        self._t = 0

    @property
    def dt(self):
        return self.env.dt

    def get_single_cost(self):
        # self._target_pos[0] and self._target_pos[2] should be close to zero (up to errors due to numerical precision)
        pos = self.env.get_relative_target_position()
        x_error = pos[0] / pos[1] - self._target_pos[0] / self._target_pos[1]
        y_error = pos[2] / pos[1] - self._target_pos[2] / self._target_pos[1]
        z_error = 1.0 / pos[1] - 1.0 / self._target_pos[1]
        return np.linalg.norm([x_error, y_error, z_error])

    def step(self, action):
        obs, reward, done, info = super(SimpleQuadPanda3dServoingEnv, self).step(action)
        if reward is None:
            self._t += 1
            done = done or \
                self._t >= self.max_time_steps or \
                not self.env.is_in_view() or \
                np.linalg.norm(self.env.get_relative_target_position()) < self.distance_threshold
            reward = - self.get_single_cost()
            if done:
                reward *= self.max_time_steps - self._t + 1
        return obs, reward, done, info

    def reset(self, state=None):
        obs = self.env.reset(state=state)
        if hasattr(self.env, 'use_car_dynamics') and self.env.use_car_dynamics:
            self.env.use_car_dynamics = False
            target_obs = self.env.observe()  # target observation should have the car in the original position
            self.env.use_car_dynamics = True
        else:
            target_obs = obs
        self._target_obs = {'target_' + k: v for (k, v) in target_obs.items()}
        self._target_pos = self.env.get_relative_target_position()
        self._t = 0
        obs.update(self._target_obs)
        return obs

    def get_state(self):
        return np.append(self.env.get_state(), self._t)

    def set_state(self, state):
        self.env.set_state(state[:-1])
        self._t = state[-1]
