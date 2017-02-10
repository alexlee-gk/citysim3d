import numpy as np
import itertools
from citysim3d.envs import SimpleQuadPanda3dEnv, Panda3dMaskCameraSensor
from citysim3d.spaces import BoxSpace
from citysim3d.utils.panda3d_util import extrude_depth, xy_to_points2d


def get_bounding_box(mask_image):
    rows = np.any(mask_image, axis=1)
    cols = np.any(mask_image, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array([cmin, rmin]), np.array([cmax, rmax])


class BboxSimpleQuadPanda3dEnv(SimpleQuadPanda3dEnv):
    def __init__(self, *args, **kwargs):
        """use_car_dynamics only applies to points"""
        self.use_car_dynamics = kwargs.pop('use_car_dynamics', False)
        super(BboxSimpleQuadPanda3dEnv, self).__init__(*args, **kwargs)

        self.mask_camera_sensor = Panda3dMaskCameraSensor(self.app, (self.skybox_node, self.city_node),
                                                          size=self.camera_size,
                                                          hfov=self.camera_hfov)
        for cam in self.mask_camera_sensor.cam:
            cam.reparentTo(self.camera_node)
        lens = self.mask_camera_sensor.cam[0].node().getLens()
        self._observation_space.spaces['points'] = BoxSpace(np.array([[-np.inf, lens.getNear(), -np.inf]] * 4),
                                                            np.array([[np.inf, lens.getFar(), np.inf]] * 4))

    def step(self, action):
        obs, reward, done, info = super(BboxSimpleQuadPanda3dEnv, self).step(action)
        done = done or obs.get('points') is None
        return obs, reward, done, info

    def observe(self):
        if self.use_car_dynamics:
            # save state and step car
            random_state = np.random.get_state()
            state = self.get_state()
            car_action = self.car_env.action_space.sample()
            self.car_env.step(car_action)

        mask, depth_image, _ = self.mask_camera_sensor.observe()
        # if the target object is not in the view, return None for the corner points
        if np.any(mask):
            min_xy, max_xy = np.array(get_bounding_box(mask))
            corners_xy = []
            for corner_xy in itertools.product(*zip(min_xy, max_xy)):
                corners_xy.append(corner_xy)
            corners_xy = np.array(corners_xy)
            # normalize between -1.0 and 1.0
            corners_2d = xy_to_points2d(self.camera_sensor.lens, corners_xy)
            # append z depth to it
            corners_z = [depth_image[corner_xy[1], corner_xy[0]] for corner_xy in corners_xy]
            corners_2d = np.c_[corners_2d, corners_z]
            # extrude to 3d points in the camera's local frame
            corners_XYZ = extrude_depth(self.camera_sensor.lens, corners_2d)
        else:
            corners_XYZ = None

        if self.use_car_dynamics:
            # restore state
            self.set_state(state)
            np.random.set_state(random_state)

        obs = super(BboxSimpleQuadPanda3dEnv, self).observe()
        obs['points'] = corners_XYZ
        return obs
