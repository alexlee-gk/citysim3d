import numpy as np
import itertools
from panda3d.core import Point3, BoundingBox
from citysim3d.envs import SimpleQuadPanda3dEnv
from citysim3d.spaces import BoxSpace, TupleSpace


class Bbox3dSimpleQuadPanda3dEnv(SimpleQuadPanda3dEnv):
    def __init__(self, *args, **kwargs):
        super(Bbox3dSimpleQuadPanda3dEnv, self).__init__(*args, **kwargs)

        if len(self.sensor_names) == 0:
            observation_spaces = []
        elif len(self.sensor_names) == 1:
            observation_spaces = [self._observation_space]
        else:
            assert isinstance(self._observation_space, TupleSpace)
            observation_spaces = list(self._observation_space.spaces)
        bbox_space = BoxSpace(-np.inf, np.inf, shape=(3,))
        self._observation_space = TupleSpace([bbox_space] + observation_spaces)

        self.car_env._car_local_node.showTightBounds()

    def observe(self):
        obj_node = self.car_env._car_local_node
        cam_node = self.camera_node
        bounds = BoundingBox(*obj_node.getTightBounds())
        corners_XYZ = np.array(list(itertools.product(*zip(bounds.getMin(), bounds.getMax()))))
        obj_to_cam_T = obj_node.getParent().getMat(cam_node)
        corners_XYZ = np.array([obj_to_cam_T.xform(Point3(*corner_XYZ)) for corner_XYZ in corners_XYZ])[:, :3]

        if self.sensor_names:
            obs = [corners_XYZ] + list(self.camera_sensor.observe())
        else:
            obs = [corners_XYZ]
        return tuple(obs)
