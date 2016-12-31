import numpy as np
from citysim3d.envs import Env
from citysim3d.envs import Panda3dMaskCameraSensor
from citysim3d.spaces import DictSpace


class ServoingEnv(Env):
    def __init__(self, env, max_time_steps=100):
        self.__dict__.update(env.__dict__)
        self.env = env
        # only non-target observation spaces
        assert isinstance(self.env.observation_space, DictSpace)
        self._observation_space = self.env.observation_space
        self._target_obs = None
        self._t = 0
        self._T = max_time_steps
        if not hasattr(self, 'mask_camera_sensor'):
            self.mask_camera_sensor = Panda3dMaskCameraSensor(self.app, (self.skybox_node, self.city_node),
                                                              size=self.camera_sensor.size,
                                                              near_far=(self.camera_sensor.lens.getNear(), self.camera_sensor.lens.getFar()),
                                                              hfov=self.camera_sensor.lens.getFov())
        for cam in self.mask_camera_sensor.cam:
            cam.reparentTo(self.camera_sensor.cam)

    def get_image_formation_error(self):
        # target relative to camera
        camera_to_target_T = np.array(self.car_node.getTransform(self.camera_node).getMat()).T
        target_direction = - camera_to_target_T[:3, 3]
        x_error = (target_direction[0] / target_direction[1])
        y_error = (target_direction[2] / target_direction[1])
        z_error = (1.0 / np.linalg.norm(target_direction) - 1.0 / np.linalg.norm(self.offset))
        # use this focal length for compatibility with the original experiments
        fov_y = np.pi / 4.
        height = 480
        focal_length = height / (2. * np.tan(fov_y / 2.))
        # TODO: the experiments uses the norm of target_direction but the paper uses the z-value of it
        # TODO: use the actual target_direction of the first time step instead of assuming anything about it
        # TODO: use the action focal length
        # z_error = 1.0 / target_direction[1] - 1.0 / np.linalg.norm(self.offset))
        # focal_length = self.camera_sensor.focal_length
        return focal_length * np.linalg.norm([x_error, y_error, z_error])

    def is_in_view(self):
        mask = self.mask_camera_sensor.observe()[0]
        return np.any(mask)

    def _step(self, action):
        return self.env.step(action)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        assert isinstance(obs, dict)
        if reward is None:
            camera_to_target_T = np.array(self.car_node.getTransform(self.camera_node).getMat()).T
            target_direction = - camera_to_target_T[:3, 3]
            self._t += 1
            done = done or \
                obs.get('points') is None or \
                self._t >= self._T or \
                not self.is_in_view() or \
                np.linalg.norm(target_direction) < 4.0
            reward = - self.get_image_formation_error()
            if done:
                reward *= self._T - self._t + 1
        obs.update(self._target_obs)
        return obs, reward, done, info

    def reset(self, state=None):
        obs = self.env.reset(state=state)
        assert isinstance(obs, dict)
        self._target_obs = {'target_' + k: v for (k, v) in obs.items()}
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
