import citysim3d.utils.transformations as tf
import numpy as np
from citysim3d.envs import Panda3dEnv, GeometricCarPanda3dEnv, Panda3dCameraSensor
from citysim3d.spaces import BoxSpace, DictSpace
from panda3d.core import AmbientLight, PointLight


class SimpleQuadPanda3dEnv(Panda3dEnv):
    def __init__(self, action_space, sensor_names=None, camera_size=None, camera_hfov=None,
                 offset=None, car_env_class=None, car_action_space=None, car_model_names=None,
                 root_node=None, dt=None):
        super(SimpleQuadPanda3dEnv, self).__init__(root_node=root_node, dt=dt)
        self._action_space = action_space
        self._sensor_names = sensor_names if sensor_names is not None else ['image']  # don't override empty list
        self._camera_size = camera_size
        self._camera_hfov = camera_hfov
        self.offset = np.array(offset) if offset is not None \
            else np.array([0, -1 / np.tan(np.pi / 6), 1]) * 15.05  # offset to quad
        self.car_env_class = car_env_class or GeometricCarPanda3dEnv
        self.car_action_space = car_action_space or BoxSpace(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        self.car_model_names = car_model_names or ['camaro2']
        self.car_env = self.car_env_class(self.car_action_space, sensor_names=[], model_names=self.car_model_names, root_node=self.root_node, dt=self.dt)
        self.skybox_node = self.car_env.skybox_node
        self.city_node = self.car_env.city_node
        self.car_node = self.car_env.car_node

        # modify the car's speed limits so that the car's speed is always a quarter of the quad's maximum forward velocity
        self.car_env.speed_offset_space.low[0] = self.action_space.high[1] / 4  # meters per second
        self.car_env.speed_offset_space.high[0] = self.action_space.high[1] / 4

        # show the quad model only if render() is called, otherwise keep it hidden
        self._load_quad()
        self.inertial_node = self.quad_node
        self.quad_node.setName('quad')
        self.quad_node.hide()
        for quad_prop_local_node in self.quad_prop_local_nodes:
            quad_prop_local_node.hide()
        self.prop_angle = 0.0
        self.prop_rpm = 10212

        camera_to_inertial_pos = np.array([0, -1 / np.tan(np.pi / 6), 1]) * -0.05  # slightly in front of the quad
        camera_to_inertial_quat = tf.quaternion_about_axis(-np.pi / 6, np.array([1, 0, 0]))
        self.camera_to_inertial_T = tf.pose_matrix(camera_to_inertial_quat, camera_to_inertial_pos)

        observation_spaces = dict()
        if self.sensor_names:
            color = depth = False
            for sensor_name in self.sensor_names:
                if sensor_name == 'image':
                    color = True
                elif sensor_name == 'depth_image':
                    depth = True
                else:
                    raise ValueError('Unknown sensor name %s' % sensor_name)
            self.camera_sensor = Panda3dCameraSensor(self.app, self.root_node, color=color, depth=depth, size=self.camera_size, hfov=self.camera_hfov)

            lens = self.camera_sensor.cam.node().getLens()
            film_size = tuple(int(s) for s in lens.getFilmSize())
            for sensor_name in self.sensor_names:
                if sensor_name == 'image':
                    observation_spaces[sensor_name] = BoxSpace(0, 255, shape=film_size[::-1] + (3,), dtype=np.uint8)
                elif sensor_name == 'depth_image':
                    observation_spaces[sensor_name] = BoxSpace(lens.getNear(), lens.getFar(), shape=film_size[::-1] + (1,))
        else:
            # still create camera sensor for functions that use camera information (e.g. isInView())
            self.camera_sensor = Panda3dCameraSensor(self.app, self.root_node, size=self.camera_size, hfov=self.camera_hfov)
        self.camera_node = self.camera_sensor.cam
        self.camera_node.setName('quad_camera')
        self.camera_node.reparentTo(self.quad_node)
        self.camera_node.setPosQuat(tuple(camera_to_inertial_pos), tuple(camera_to_inertial_quat))
        self._observation_space = DictSpace(observation_spaces)
        self._first_render = True

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def sensor_names(self):
        return self._sensor_names

    @property
    def camera_size(self):
        return self._camera_size

    @property
    def camera_hfov(self):
        return self._camera_hfov

    def _load_quad(self):
        self.quad_node = self.app.loader.loadModel('iris')
        self.quad_node.reparentTo(self.root_node)

        ambient_color = (.1, .1, .1, 1)
        sun_light_color = (.5, .5, .5, 1)
        self.quad_node.setLightOff()
        ambient_light = self.root_node.attachNewNode(AmbientLight('quad_ambient_light'))
        ambient_light.node().setColor(ambient_color)
        self.quad_node.setLight(ambient_light)
        sun_light = self.root_node.attachNewNode(PointLight('quad_sun_light'))
        sun_light.node().setColor(sun_light_color)
        sun_light.setPos((-2506., -634., 2596.))
        self.quad_node.setLight(sun_light)
        self.quad_node.flattenStrong()

        quad_prop_positions = [np.array([ 0.20610,  0.13830, 0.025]),  # blue, right
                               np.array([ 0.22254, -0.12507, 0.025]),  # black, right
                               np.array([-0.20266,  0.13830, 0.025]),  # blue, left
                               np.array([-0.21911, -0.12507, 0.025])]  # black, left
        self.quad_prop_local_nodes = []
        for quad_prop_id, quad_prop_pos in enumerate(quad_prop_positions):
            is_ccw = quad_prop_id in (1, 2)
            quad_prop_node = self.quad_node.attachNewNode('quad_prop_%d' % quad_prop_id)
            quad_prop_local_node = self.app.loader.loadModel('iris_prop_%s' % ('ccw' if is_ccw else 'cw'))
            quad_prop_local_node.reparentTo(quad_prop_node)
            quad_prop_node.setPos(tuple(quad_prop_pos))
            quad_prop_node.flattenStrong()
            self.quad_prop_local_nodes.append(quad_prop_local_node)

    def step(self, action):
        # update the angle of the propellers (for rendering purposes)
        if self.prop_rpm:
            self.prop_angle += (self.prop_rpm * 2 * np.pi / 60) * self.dt
            self.prop_angle -= 2 * np.pi * np.floor(self.prop_angle / (2 * np.pi))
            for quad_prop_id, quad_prop_local_node in enumerate(self.quad_prop_local_nodes):
                is_ccw = quad_prop_id in (1, 2)
                angle = self.prop_angle if is_ccw else -self.prop_angle
                quad_prop_local_node.setQuat(tuple(tf.quaternion_about_axis(angle, np.array([0, 0, 1]))))

        car_action = self.car_env.action_space.sample()
        self.car_env.step(car_action)

        # update action to be within the action space
        if not self.action_space.contains(action):
            action = self.action_space.clip(action, out=action)

        linear_vel, angular_vel = np.split(action, [3])
        if angular_vel.shape == (1,):
            angular_vel = angular_vel * self._action_space.axis

        # compute next state
        quad_T = tf.pose_matrix(self.quad_node.getQuat(), self.quad_node.getPos())
        quad_to_next_quad_T = tf.position_axis_angle_matrix(np.append(linear_vel, angular_vel) * self.dt)
        next_quad_T = quad_T.dot(quad_to_next_quad_T)

        # set new state
        self.quad_node.setPosQuat(tuple(next_quad_T[:3, 3]), tuple(tf.quaternion_from_matrix(next_quad_T[:3, :3])))

        # update action to be consistent with the state clippings
        quad_to_next_quad_T = tf.inverse_matrix(quad_T).dot(next_quad_T)

        linear_vel, angular_vel = np.split(tf.position_axis_angle_from_matrix(quad_to_next_quad_T) / self.dt, [3])
        # project angular_vel onto the axis
        if self._action_space.axis is not None:
            angular_vel = angular_vel.dot(self._action_space.axis)
        action[:] = np.append(linear_vel, angular_vel)

        return self.observe(), None, False, dict()

    def reset(self, state=None):
        self._first_render = True
        if state is None:
            self.car_env.reset()
            # don't set the car's state again since its reset already did it
            quad_pos, quad_quat = self.compute_desired_quad_pos_quat()
            self.quad_node.setPosQuat(tuple(quad_pos), tuple(quad_quat))
        else:
            self.set_state(state)
        return self.observe()

    @property
    def hor_car_T(self):
        hor_car_T = tf.pose_matrix(self.car_node.getQuat(), self.car_node.getPos())
        up = np.array([0, 0, 1])
        angle = tf.angle_between_vectors(hor_car_T[:3, 2], up)
        if angle != 0.0:
            axis = np.cross(hor_car_T[:3, 2], up)
            project_T = tf.rotation_matrix(angle, axis, point=hor_car_T[:3, 3])
            hor_car_T = project_T.dot(hor_car_T)
        return hor_car_T

    def compute_desired_quad_pos_quat(self, offset=None):
        offset = offset if offset is not None else self.offset
        # desired position of the quad is located at offset relative to the car (reoriented so that the car is horizontal)
        hor_car_T = self.hor_car_T
        des_quad_T = hor_car_T.dot(tf.translation_matrix(offset))
        # adjust desired rotation of the quad so that it points towards the car while constraining the z-axis to be up
        up = np.array([0, 0, 1])
        des_quat_rot_y = hor_car_T[:3, 3] - des_quad_T[:3, 3]
        des_quat_rot_y -= des_quat_rot_y.dot(up) * up  # project to horizontal plane
        angle = tf.angle_between_vectors(des_quad_T[:3, 1], des_quat_rot_y)
        if angle != 0.0:
            axis = np.cross(des_quad_T[:3, 1], des_quat_rot_y)
            project_T = tf.rotation_matrix(angle, axis, point=des_quad_T[:3, 3])
            des_quad_T = project_T.dot(des_quad_T)
        des_quad_pos = des_quad_T[:3, 3]
        des_quad_quat = tf.quaternion_from_matrix(des_quad_T[:3, :3])
        return des_quad_pos, des_quad_quat

    def get_state(self):
        quad_T = tf.pose_matrix(self.quad_node.getQuat(), self.quad_node.getPos())
        quad_state = tf.position_axis_angle_from_matrix(quad_T)
        car_state = self.car_env.get_state()
        return np.concatenate([quad_state, car_state])

    def set_state(self, state):
        quad_state, car_state = np.split(state, [6])
        self.car_env.set_state(car_state)
        quad_T = tf.position_axis_angle_matrix(quad_state)
        quad_pos = quad_T[:3, 3]
        quad_quat = tf.quaternion_from_matrix(quad_T[:3, :3])
        self.quad_node.setPosQuat(tuple(quad_pos), tuple(quad_quat))

    def observe(self):
        if self.sensor_names:
            return dict(zip(self.sensor_names, self.camera_sensor.observe()))
        else:
            return dict()

    def render(self):
        if self.quad_node.isHidden():
            self.quad_node.show()
            for quad_prop_local_node in self.quad_prop_local_nodes:
                quad_prop_local_node.show()

        if self._first_render:
            tightness = 1.0
            self._first_render = False
        else:
            tightness = 0.1
        target_node = self.quad_node
        target_T = tf.pose_matrix(target_node.getQuat(), target_node.getPos())
        target_camera_pos = target_T[:3, 3] + target_T[:3, :3].dot(np.array([0., -1 / np.tan(np.pi / 6), 1]) * 15)
        self.app.cam.setPos(tuple((1 - tightness) * np.array(self.app.cam.getPos()) + tightness * target_camera_pos))
        self.app.cam.lookAt(target_node)

        # render observation window(s)
        for _ in range(self.camera_sensor.graphics_engine.getNumWindows()):
            self.camera_sensor.graphics_engine.renderFrame()
        self.camera_sensor.graphics_engine.syncFrame()

        # render main window(s)
        for _ in range(self.app.graphicsEngine.getNumWindows()):
            self.app.graphicsEngine.renderFrame()
        self.app.graphicsEngine.syncFrame()

    def get_relative_target_position(self):
        return np.array(self.car_node.getTransform(self.camera_node).getPos())

    def is_in_view(self):
        return self.camera_node.node().isInView(self.car_node.getTransform(self.camera_node).getPos())
