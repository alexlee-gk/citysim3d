import itertools
import os

import numpy as np
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

        self.mask_camera_sensor = Panda3dMaskCameraSensor(self.app, self.root_node, (self.skybox_node, self.city_node),
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


class CcotBboxSimpleQuadPanda3dEnv(BboxSimpleQuadPanda3dEnv):
    """
    This class uses bounding boxes computed by the MATLAB implementation of
    the Continuous Convolution Operator Tracker (C-COT):
    https://github.com/martin-danelljan/Continuous-ConvOp

    The official matlab package or the transplant package can be used to
    interface with MATLAB. We use the latter one since it's about twice as
    fast.
    """
    def __init__(self, *args, **kwargs):
        super(CcotBboxSimpleQuadPanda3dEnv, self).__init__(*args, **kwargs)
        ccot_path = os.environ.get('CCOT_DIR', os.path.expanduser('~/rll/Continuous-ConvOp'))

        # import matlab.engine
        # self.eng = matlab.engine.start_matlab()
        # self.eng.addpath(ccot_path)
        # self.eng.setup_paths(nargout=0)

        import transplant
        self.matlab = transplant.Matlab()
        self.matlab.addpath(ccot_path)
        self.matlab.setup_paths(nargout=0)

        self.tracker = None

    def reset(self, state=None):
        self.tracker = None
        return super(CcotBboxSimpleQuadPanda3dEnv, self).reset(state=state)

    def init_tracker(self, image, bbox):
        bbox_min, bbox_max = bbox
        left, top = bbox_min
        width, height = bbox_max - bbox_min

        wsize = np.array([height, width])
        init_pos = np.array([top, left]) + wsize // 2

        # import matlab
        # im = matlab.uint8(image.tolist())
        # init_pos = matlab.double(init_pos.tolist())
        # wsize = matlab.double(wsize.tolist())
        # self.tracker = self.eng.Tracker(im, init_pos, wsize, 100)

        init_pos = init_pos.astype(np.float)
        wsize = wsize.astype(np.float)
        self.tracker = self.matlab.Tracker(image, init_pos, wsize, 100)

    def get_tracker_bounding_box(self, image):
        # import matlab
        # im = matlab.uint8(image.tolist())
        # rect_position = self.eng.step(self.tracker, im)

        rect_position = self.matlab.step(self.tracker, image, nargout=1)
        left, top, width, height = np.array(rect_position[0], dtype=int)
        return np.array([left, top]), np.array([left + width, top + height])

    def bounding_box_to_XYZ(self, depth_image, bbox):
        min_xy, max_xy = bbox
        min_xy = np.maximum(min_xy, 0)
        max_xy = np.minimum(max_xy, np.asarray(self.camera_size) - 1)
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
        return corners_XYZ

    def observe(self):
        if self.use_car_dynamics:
            # save state and step car
            random_state = np.random.get_state()
            state = self.get_state()
            car_action = self.car_env.action_space.sample()
            self.car_env.step(car_action)

        mask, depth_image, _ = self.mask_camera_sensor.observe()
        image = self.camera_sensor.observe()[0]
        if self.tracker is None:
            ground_truth_bbox = get_bounding_box(mask)
            self.init_tracker(image, ground_truth_bbox)
        corners_XYZ = self.bounding_box_to_XYZ(depth_image, self.get_tracker_bounding_box(image))

        if self.use_car_dynamics:
                # restore state
            self.set_state(state)
            np.random.set_state(random_state)

        obs = SimpleQuadPanda3dEnv.observe(self)
        obs['points'] = corners_XYZ
        return obs
