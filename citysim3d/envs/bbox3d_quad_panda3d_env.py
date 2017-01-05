import numpy as np
import itertools
from panda3d.core import Point3, BoundingBox
from citysim3d.envs import SimpleQuadPanda3dEnv
from citysim3d.spaces import BoxSpace, TupleSpace


class Bbox3dSimpleQuadPanda3dEnv(SimpleQuadPanda3dEnv):
    def __init__(self, *args, **kwargs):
        super(Bbox3dSimpleQuadPanda3dEnv, self).__init__(*args, **kwargs)

        self._observation_space.spaces['points'] = BoxSpace(-np.inf, np.inf, shape=(3,))

        # uncomment to visualize bounding box (makes rendering slow)
        # self.car_env._car_local_node.showTightBounds()

    def observe(self):
        obj_node = self.car_env._car_local_node
        cam_node = self.camera_node
        bounds = BoundingBox(*obj_node.getTightBounds())
        corners_XYZ = np.array(list(itertools.product(*zip(bounds.getMin(), bounds.getMax()))))
        obj_to_cam_T = obj_node.getParent().getMat(cam_node)
        corners_XYZ = np.array([obj_to_cam_T.xform(Point3(*corner_XYZ)) for corner_XYZ in corners_XYZ])[:, :3]

        obs = super(Bbox3dSimpleQuadPanda3dEnv, self).observe()
        obs['points'] = corners_XYZ
        return obs
