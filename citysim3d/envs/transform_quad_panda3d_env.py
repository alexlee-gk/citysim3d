import numpy as np
from citysim3d.envs import SimpleQuadPanda3dEnv
from citysim3d.utils import transformations as tf


class TransformSimpleQuadPanda3dEnv(SimpleQuadPanda3dEnv):
    def __init__(self, *args, **kwargs):
        """use_car_dynamics only applies to transform"""
        self.use_car_dynamics = kwargs.pop('use_car_dynamics', False)
        super(TransformSimpleQuadPanda3dEnv, self).__init__(*args, **kwargs)

        self._observation_space.spaces['transform'] = None  # TODO

    def observe(self):
        if self.use_car_dynamics:
            # save state and step car
            random_state = np.random.get_state()
            state = self.get_state()
            car_action = self.car_env.action_space.sample()
            self.car_env.step(car_action)

        curr_to_obj_T = tf.inverse_matrix(np.array(self.quad_node.getTransform().getMat()).T).dot(self.hor_car_T)

        if self.use_car_dynamics:
            # restore state
            self.set_state(state)
            np.random.set_state(random_state)

        obs = super(TransformSimpleQuadPanda3dEnv, self).observe()
        obs['transform'] = curr_to_obj_T
        return obs
