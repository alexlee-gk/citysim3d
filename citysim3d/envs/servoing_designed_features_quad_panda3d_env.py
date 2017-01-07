import numpy as np
import cv2
from citysim3d.envs import SimpleQuadPanda3dEnv, ServoingEnv
from citysim3d.spaces import BoxSpace
from citysim3d.utils.panda3d_util import xy_depth_to_XYZ


def is_present(point_xy, mask):
    """
    Returns true if the pixel at the coordinates point_xy is non-zero in the
    mask. If point_xy are floating-point coordinates, this function returns
    true if any of the pixels that contains the coordinates are non-zero.
    """
    for x in range(int(np.floor(point_xy[0])), int(np.ceil(point_xy[0]) + 1)):
        for y in range(int(np.floor(point_xy[1])), int(np.ceil(point_xy[1]) + 1)):
            if mask[y, x]:
                return True
    return False


class ServoingDesignedFeaturesSimpleQuadPanda3dEnv(SimpleQuadPanda3dEnv, ServoingEnv):
    def __init__(self, action_space, sensor_names=None, offset=None,
                 feature_type=None, filter_features=None, max_time_steps=100,
                 car_env_class=None, car_action_space=None, car_model_names=None,
                 app=None, dt=None):
        super(ServoingDesignedFeaturesSimpleQuadPanda3dEnv, self).__init__(action_space,
                                                                           sensor_names=sensor_names,
                                                                           offset=offset,
                                                                           car_env_class=car_env_class,
                                                                           car_action_space=car_action_space,
                                                                           car_model_names=car_model_names,
                                                                           app=app, dt=dt)

        self._observation_space.spaces['points'] = BoxSpace(np.array([-np.inf, self.quad_camera_node.node().getLens().getNear(), -np.inf]),
                                                            np.array([np.inf, self.quad_camera_node.node().getLens().getFar(), np.inf]))

        self.filter_features = True if filter_features is None else False
        self._feature_type = feature_type or 'sift'
        if cv2.__version__.split('.')[0] == '3':
            from cv2.xfeatures2d import SIFT_create, SURF_create
            from cv2 import ORB_create
            if self.feature_type == 'orb':
                # https://github.com/opencv/opencv/issues/6081
                cv2.ocl.setUseOpenCL(False)
        else:
            SIFT_create = cv2.SIFT
            SURF_create = cv2.SURF
            ORB_create = cv2.ORB
        if self.feature_type == 'sift':
            self._feature_extractor = SIFT_create()
        elif self.feature_type == 'surf':
            self._feature_extractor = SURF_create()
        elif self.feature_type == 'orb':
            self._feature_extractor = ORB_create()
        else:
            raise ValueError("Unknown feature extractor %s" % self.feature_type)
        if self.feature_type == 'orb':
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self._target_key_points = None
        self._target_descriptors = None

        self._t = 0
        self._T = max_time_steps

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def target_key_points(self):
        return self._target_key_points

    @property
    def target_descriptors(self):
        return self._target_descriptors

    def _step(self, action):
        return SimpleQuadPanda3dEnv.step(self, action)

    def step(self, action):
        return ServoingEnv.step(self, action)

    def reset(self, state=None):
        self._target_obs = None
        self._target_key_points = None
        self._target_descriptors = None
        self._t = 0
        return super(ServoingDesignedFeaturesSimpleQuadPanda3dEnv, self).reset(state=state)

    def observe(self):
        obs = super(ServoingDesignedFeaturesSimpleQuadPanda3dEnv, self).observe()
        assert isinstance(obs, dict)

        mask, depth_image, _ = self.mask_camera_sensor.observe()
        if self._target_obs is None:
            self._target_obs = {'target_' + k: v for (k, v) in obs.items()}
            key_points, descriptors = self._feature_extractor.detectAndCompute(self._target_obs['target_image'], None)
            self._target_key_points = []
            self._target_descriptors = []
            for key_point, descriptor in zip(key_points, descriptors):
                if is_present(key_point.pt, mask):
                    self._target_key_points.append(key_point)
                    self._target_descriptors.append(descriptor)
            self._target_descriptors = np.asarray(self._target_descriptors)

        key_points, descriptors = self._feature_extractor.detectAndCompute(obs['image'], None)
        matches = self._matcher.match(descriptors, self._target_descriptors)

        if self.filter_features:
            matches = [match for match in matches if is_present(key_points[match.queryIdx].pt, mask)]

        if matches:
            key_points_xy = np.array([key_points[match.queryIdx].pt for match in matches])
            target_key_points_xy = np.array([self._target_key_points[match.trainIdx].pt for match in matches])
            key_points_XYZ = xy_depth_to_XYZ(self.camera_sensor.lens, key_points_xy, depth_image)
            target_key_points_XYZ = xy_depth_to_XYZ(self.camera_sensor.lens, target_key_points_xy, depth_image)
            obs['points'] = key_points_XYZ
            obs['target_points'] = target_key_points_XYZ

        obs.update(self._target_obs)
        return obs
