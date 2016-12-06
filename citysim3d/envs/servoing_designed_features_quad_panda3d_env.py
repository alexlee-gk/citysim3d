import numpy as np
import cv2
from citysim3d.envs import SimpleQuadPanda3dEnv
from citysim3d.envs import Panda3dMaskCameraSensor
from citysim3d.spaces import BoxSpace, TupleSpace
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


class ServoingDesignedFeaturesSimpleQuadPanda3dEnv(SimpleQuadPanda3dEnv):
    def __init__(self, action_space, sensor_names=None, offset=None,
                 feature_type=None, filter_features=None, max_time_steps=100,
                 car_env_class=None, car_action_space=None, car_model_name=None,
                 app=None, dt=None):
        super(ServoingDesignedFeaturesSimpleQuadPanda3dEnv, self).__init__(action_space,
                                                                           sensor_names=sensor_names,
                                                                           offset=offset,
                                                                           car_env_class=car_env_class,
                                                                           car_action_space=car_action_space,
                                                                           car_model_name=car_model_name,
                                                                           app=app, dt=dt)

        if len(self.sensor_names) == 0:
            observation_spaces = []
        elif len(self.sensor_names) == 1:
            observation_spaces = [self._observation_space]
        else:
            assert isinstance(self._observation_space, TupleSpace)
            observation_spaces = list(self._observation_space.spaces)
        bbox_space = BoxSpace(np.array([-np.inf, self.quad_camera_node.node().getLens().getNear(), -np.inf]),
                              np.array([np.inf, self.quad_camera_node.node().getLens().getFar(), np.inf]))
        self._observation_space = TupleSpace([bbox_space] + observation_spaces)

        self.mask_camera_sensor = Panda3dMaskCameraSensor(self.app, (self.skybox_node, self.city_node),
                                                          size=self.camera_sensor.size,
                                                          near_far=(self.camera_sensor.lens.getNear(), self.camera_sensor.lens.getFar()),
                                                          hfov=self.camera_sensor.lens.getFov())
        for cam in self.mask_camera_sensor.cam:
            cam.reparentTo(self.camera_sensor.cam)

        self.filter_features = True if filter_features is None else False
        self._feature_type = None or 'sift'
        if self.feature_type == 'sift':
            self._feature_extractor = cv2.ORB()
        elif self.feature_type == 'surf':
            self._feature_extractor = cv2.SURF()
        elif self.feature_type == 'orb':
            self._feature_extractor = cv2.ORB()
        else:
            raise ValueError("Unknown feature extractor %s" % self.feature_type)
        if self.feature_type == 'orb':
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self._target_image = None
        self._target_key_points = None
        self._target_descriptors = None

        self._t = 0
        self._T = max_time_steps

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def target_image(self):
        return self._target_image

    @property
    def target_key_points(self):
        return self._target_key_points

    @property
    def target_descriptors(self):
        return self._target_descriptors

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

    def step(self, action):
        obs, reward, done, info = super(ServoingDesignedFeaturesSimpleQuadPanda3dEnv, self).step(action)
        if obs is None or obs[0] is None:  # observation for the corner points is None
            done = True
        camera_to_target_T = np.array(self.car_node.getTransform(self.camera_node).getMat()).T
        target_direction = - camera_to_target_T[:3, 3]
        self._t += 1
        done = done or \
               self._t >= self._T or \
               not self.is_in_view() or \
               np.linalg.norm(target_direction) < 4.0
        reward = - self.get_image_formation_error()
        if done:
            reward *= self._T - self._t + 1
        return obs, reward, done, info

    def reset(self, state=None):
        self._target_image = None
        self._target_obs = None
        self._target_key_points = None
        self._target_descriptors = None
        self._t = 0
        return super(ServoingDesignedFeaturesSimpleQuadPanda3dEnv, self).reset(state=state)

    def observe(self):
        obs = self.camera_sensor.observe()
        if len(obs) == 1:
            image, = obs
        elif len(obs) == 2:
            image, _ = obs
        else:
            raise ValueError("The observation contains no images.")

        mask, depth_image, _ = self.mask_camera_sensor.observe()
        if self._target_image is None:
            self._target_obs = obs
            self._target_image = image
            key_points, descriptors = self._feature_extractor.detectAndCompute(image, None)
            self._target_key_points = []
            self._target_descriptors = []
            for key_point, descriptor in zip(key_points, descriptors):
                if is_present(key_point.pt, mask):
                    self._target_key_points.append(key_point)
                    self._target_descriptors.append(descriptor)
            self._target_descriptors = np.asarray(self._target_descriptors)

        key_points, descriptors = self._feature_extractor.detectAndCompute(image, None)
        matches = self._matcher.match(descriptors, self._target_descriptors)

        if self.filter_features:
            matches = [match for match in matches if is_present(key_points[match.queryIdx].pt, mask)]
        if not matches:
            return None

        key_points_xy = np.array([key_points[match.queryIdx].pt for match in matches])
        target_key_points_xy = np.array([self._target_key_points[match.trainIdx].pt for match in matches])
        key_points_XYZ = xy_depth_to_XYZ(self.camera_sensor.lens, key_points_xy, depth_image)
        target_key_points_XYZ = xy_depth_to_XYZ(self.camera_sensor.lens, target_key_points_xy, depth_image)

        obs = [key_points_XYZ] + list(obs) + [target_key_points_XYZ] + list(self._target_obs)
        return tuple(obs)
