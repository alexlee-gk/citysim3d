import numpy as np
import cv2
from citysim3d.envs import SimpleQuadPanda3dEnv, ServoingEnv, Panda3dMaskCameraSensor
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
    def __init__(self, action_space, feature_type=None, filter_features=None,
                 max_time_steps=100, distance_threshold=4.0, **kwargs):
        """
        filter_features indicates whether to filter out key points that are not
        on the object in the current image. Key points in the target image are
        always filtered out.
        """
        SimpleQuadPanda3dEnv.__init__(self, action_space, **kwargs)
        ServoingEnv.__init__(self, env=self, max_time_steps=max_time_steps, distance_threshold=distance_threshold)

        lens = self.camera_node.node().getLens()
        self._observation_space.spaces['points'] = BoxSpace(np.array([-np.inf, lens.getNear(), -np.inf]),
                                                            np.array([np.inf, lens.getFar(), np.inf]))
        film_size = tuple(int(s) for s in lens.getFilmSize())
        self.mask_camera_sensor = Panda3dMaskCameraSensor(self.app, (self.skybox_node, self.city_node),
                                                          size=film_size,
                                                          near_far=(lens.getNear(), lens.getFar()),
                                                          hfov=lens.getFov())
        for cam in self.mask_camera_sensor.cam:
            cam.reparentTo(self.camera_sensor.cam)

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
        obs, reward, done, info = SimpleQuadPanda3dEnv.step(self, action)
        done = done or obs.get('points') is None
        return obs, reward, done, info

    def step(self, action):
        return ServoingEnv.step(self, action)

    def reset(self, state=None):
        self._target_key_points = None
        self._target_descriptors = None
        self._target_obs = None
        self._target_pos = self.get_relative_target_position()
        self._t = 0
        return SimpleQuadPanda3dEnv.reset(self, state=state)

    def observe(self):
        obs = SimpleQuadPanda3dEnv.observe(self)
        assert isinstance(obs, dict)

        mask, depth_image, _ = self.mask_camera_sensor.observe()  # used to filter out key points
        if self._target_obs is None:
            self._target_obs = {'target_' + k: v for (k, v) in obs.items()}
            self._target_key_points, self._target_descriptors = \
                self._feature_extractor.detectAndCompute(self._target_obs['target_image'], mask)

        if self._target_key_points:
            key_points, descriptors = \
                self._feature_extractor.detectAndCompute(obs['image'], mask if self.filter_features else None)
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
