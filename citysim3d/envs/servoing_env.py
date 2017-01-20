import numpy as np
from citysim3d.envs import Env
from citysim3d.spaces import DictSpace


class ServoingEnv(Env):
    def __init__(self, env, max_time_steps=100, distance_threshold=4.0):
        """
        env should implement get_relative_target_position, is_in_view for this
        instance to return a reward that is not None
        """
        self.__dict__.update(env.__dict__)
        self.env = env
        # only non-target observation spaces
        assert isinstance(self.env.observation_space, DictSpace)
        self._observation_space = self.env.observation_space
        self._target_obs = None
        self._target_pos = None  # position of the target relative to the camera
        self._t = 0
        self.max_time_steps = max_time_steps
        self.distance_threshold = distance_threshold

    def get_image_formation_error(self):
        pos = self.env.get_relative_target_position()
        x_error = pos[0] / pos[1]
        y_error = pos[2] / pos[1]
        z_error = 1.0 / pos[1] - 1.0 / self._target_pos[1]
        # self._target_pos[0] and self._target_pos[2] are assumed to be zero (up to errors due to numerical precision)
        return np.linalg.norm([x_error, y_error, z_error])

    def _step(self, action):
        return self.env.step(action)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        assert isinstance(obs, dict)
        if reward is None:
            self._t += 1
            try:
                done = done or \
                    self._t >= self.max_time_steps or \
                    not self.env.is_in_view() or \
                    np.linalg.norm(self.env.get_relative_target_position()) < self.distance_threshold
                reward = - self.get_image_formation_error()
                if done:
                    reward *= self.max_time_steps - self._t + 1
            except AttributeError:
                # return None for the reward if self.env doesn't implement
                # get_relative_target_position or is_in_view
                pass
        obs.update(self._target_obs)
        return obs, reward, done, info

    def reset(self, state=None):
        obs = self.env.reset(state=state)
        assert isinstance(obs, dict)
        self._target_obs = {'target_' + k: v for (k, v) in obs.items()}
        self._target_pos = self.env.get_relative_target_position()
        self._t = 0
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
        spaces = dict(self._observation_space.spaces)
        target_spaces = {'target_' + k: space for (k, space) in self._observation_space.spaces.items()}
        spaces.update(target_spaces)
        return DictSpace(spaces)
