import cv2
import numpy as np
from citysim3d.envs import SimpleQuadPanda3dEnv, Panda3dMaskCameraSensor
from citysim3d.envs.servoing_env import SimpleQuadPanda3dServoingEnv
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


class DesignedFeaturesSimpleQuadPanda3dServoingEnv(SimpleQuadPanda3dEnv, SimpleQuadPanda3dServoingEnv):
    def __init__(self, action_space, feature_type=None, filter_features=None,
                 max_time_steps=100, distance_threshold=4.0, **kwargs):
        """
        filter_features indicates whether to filter out key points that are not
        on the object in the current image. Key points in the target image are
        always filtered out.
        """
        SimpleQuadPanda3dEnv.__init__(self, action_space, **kwargs)
        SimpleQuadPanda3dServoingEnv.__init__(self, env=self, max_time_steps=max_time_steps, distance_threshold=distance_threshold)

        self.mask_camera_sensor = Panda3dMaskCameraSensor(self.app, self.root_node, (self.skybox_node, self.city_node),
                                                          size=self.camera_size,
                                                          hfov=self.camera_hfov)
        for cam in self.mask_camera_sensor.cam:
            cam.reparentTo(self.camera_sensor.cam)
        lens = self.mask_camera_sensor.cam[0].node().getLens()
        self._observation_space.spaces['points'] = BoxSpace(np.array([[-np.inf, lens.getNear(), -np.inf]]),
                                                            np.array([[np.inf, lens.getFar(), np.inf]]))  # the number of feature points is not known in advance

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
            self._feature_extractor = SIFT_create(contrastThreshold=0.035)  # the default of 0.04 doesn't produce enough features
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

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def target_key_points(self):
        return self._target_key_points

    @property
    def target_descriptors(self):
        return self._target_descriptors

    def step(self, action):
        obs, reward, done, info = SimpleQuadPanda3dEnv.step(self, action)
        done = done or obs.get('points') is None
        if reward is None:
            self._t += 1
            done = done or \
                self._t >= self.max_time_steps or \
                not self.is_in_view() or \
                np.linalg.norm(self.get_relative_target_position()) < self.distance_threshold
            reward = - self.get_single_cost()
            if done:
                reward *= self.max_time_steps - self._t + 1
        return obs, reward, done, info

    def reset(self, state=None):
        self._target_key_points = None
        self._target_descriptors = None
        self._target_obs = None
        self._target_pos = self.get_relative_target_position()
        self._t = 0
        obs = SimpleQuadPanda3dEnv.reset(self, state=state)
        return obs

    def _detect_and_compute(self, image, mask):
        key_points, descriptors = self._feature_extractor.detectAndCompute(image, mask)
        if not key_points:
            # it seems that detectAndCompute returns less features than it
            # should, so get all the features, and then filter them
            key_points, descriptors = self._feature_extractor.detectAndCompute(image, None)
            if key_points:  # there are no features in the unmasked image
                key_points_descriptors = [key_point_descriptor for key_point_descriptor in zip(key_points, descriptors)
                                          if is_present(key_point_descriptor[0].pt, mask)]
                if key_points_descriptors:
                    key_points, descriptors = zip(*key_points_descriptors)
                    key_points = list(key_points)
                    descriptors = np.asarray(descriptors)
                else:  # all features got filtered
                    key_points = []
                    descriptors = None
        return key_points, descriptors

    def observe(self):
        obs = SimpleQuadPanda3dEnv.observe(self)

        mask, depth_image, _ = self.mask_camera_sensor.observe()  # used to filter out key points
        if self._target_obs is None:
            self._target_obs = {'target_' + k: v for (k, v) in obs.items()}
            self._target_key_points, self._target_descriptors = self._detect_and_compute(self._target_obs['target_image'], mask)

        if self._target_key_points:
            key_points, descriptors = self._detect_and_compute(obs['image'], mask if self.filter_features else None)
            matches = self._matcher.match(descriptors, self._target_descriptors)
        else:
            matches = False

        if matches:
            key_points_xy = np.array([key_points[match.queryIdx].pt for match in matches])
            target_key_points_xy = np.array([self._target_key_points[match.trainIdx].pt for match in matches])
            key_points_XYZ = xy_depth_to_XYZ(self.camera_sensor.lens, key_points_xy, depth_image)
            target_key_points_XYZ = xy_depth_to_XYZ(self.camera_sensor.lens, target_key_points_xy, depth_image)
            obs['points'] = key_points_XYZ
            obs['target_points'] = target_key_points_XYZ
        else:
            obs['points'] = None
            obs['target_points'] = None

        obs.update(self._target_obs)
        return obs
