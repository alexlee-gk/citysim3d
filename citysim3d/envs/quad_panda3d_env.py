import numpy as np
from panda3d.core import AmbientLight, PointLight
from citysim3d.envs import Panda3dEnv, Panda3dCameraSensor, GeometricCarPanda3dEnv
from citysim3d.spaces import BoxSpace, TupleSpace
import citysim3d.utils.transformations as tf
from citysim3d.utils import scale_crop_camera_parameters


class SimpleQuadPanda3dEnv(Panda3dEnv):
    def __init__(self, action_space, sensor_names=None, offset=None,
                 car_env_class=None, car_action_space=None, car_model_name=None,
                 app=None, dt=None):
        super(SimpleQuadPanda3dEnv, self).__init__(app=app, dt=dt)
        self._action_space = action_space
        self._sensor_names = sensor_names if sensor_names is not None else ['image']  # don't override empty list
        self.offset = offset if offset is not None else np.array([0, -np.sqrt(3), 1]) * 15
        self.car_env_class = car_env_class or GeometricCarPanda3dEnv
        self.car_action_space = car_action_space or BoxSpace(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        self.car_model_name = car_model_name or ['camaro2']
        if not isinstance(self.car_model_name, (tuple, list)):
            self.car_model_name = [self.car_model_name]
        self.car_env = self.car_env_class(self.car_action_space, sensor_names=[], model_names=self.car_model_name, app=self.app, dt=self.dt)
        self.skybox_node = self.car_env.skybox_node
        self.city_node = self.car_env.city_node
        self.car_node = self.car_env.car_node

        # modify the car's speed limits so that the car's speed is always a quarter of the quad's maximum forward velocity
        self.car_env.speed_offset_space.low[0] = self.action_space.high[1] / 4  # meters per second
        self.car_env.speed_offset_space.high[0] = self.action_space.high[1] / 4

        self._load_quad()
        self.quad_node.setName('quad')
        self.prop_angle = 0.0
        self.prop_rpm = 10212

        if self.sensor_names:
            # orig_size = (640, 480)
            # orig_hfov = 60.0
            # scale_size = 0.125
            # crop_size = size = (32, 32)
            # hfov = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(orig_hfov) / 2.) * crop_size[0] / orig_size[0] / scale_size))
            # # size = orig_size
            # # hfov = orig_hfov
            #
            # scale_size = 1.0
            # crop_size = size = (32 * 8, 32 * 8)
            # hfov = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(orig_hfov) / 2.) * crop_size[0] / orig_size[0] / scale_size))

            size, hfov = scale_crop_camera_parameters((640, 480), 60.0, crop_size=(int(32 / 0.125),) * 2)

            color = depth = False
            for sensor_name in self.sensor_names:
                if sensor_name == 'image':
                    color = True
                elif sensor_name == 'depth_image':
                    depth = True
                else:
                    raise ValueError('Unknown sensor name %s' % sensor_name)
            self.quad_camera_sensor = self.camera_sensor = Panda3dCameraSensor(self.app, color=color, depth=depth, size=size, hfov=hfov)
            self.quad_camera_node = self.camera_node = self.quad_camera_sensor.cam
            self.quad_camera_node.reparentTo(self.quad_node)
            self.quad_camera_node.setPos(tuple(np.array([0, -np.sqrt(3), 1]) * -0.05))  # slightly in front of the quad
            self.quad_camera_node.setQuat(tuple(tf.quaternion_about_axis(-np.pi / 6, np.array([1, 0, 0]))))
            self.quad_camera_node.setName('quad_camera')

            # # import IPython as ipy; ipy.embed()
            #
            # lens = self.quad_camera_sensor.lens
            # orig_size = lens.getFilmSize()
            # orig_hfov = lens.getFov()[0]
            #
            # scale_size = 0.125
            # crop_size = (32, 32)
            #
            # # f = (orig_size[0] / 2.) / np.tan(np.deg2rad(orig_hfov) / 2.)
            # # hfov = np.rad2deg(2 * np.arctan((crop_size[0] / 2.) / (f * scale_size)))
            #
            # hfov = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(orig_hfov) / 2.) * crop_size[0] / orig_size[0] / scale_size))
            #
            # self.quad_camera_sensor2 = self.camera_sensor = Panda3dCameraSensor(self.app, color=color, depth=depth, size=crop_size)
            # self.quad_camera_sensor2.cam.reparentTo(self.quad_camera_node)
            # lens2 = self.quad_camera_sensor2.lens
            # # lens2.setFilmSize(*crop_size)
            # lens2.setFov(hfov)

            size = self.quad_camera_sensor.size
            observation_spaces = []
            for sensor_name in self.sensor_names:
                if sensor_name == 'image':
                    observation_spaces.append(BoxSpace(0, 255, shape=(size[1], size[0], 3), dtype=np.uint8))
                elif sensor_name == 'depth_image':
                    observation_spaces.append(BoxSpace(self.quad_camera_node.node().getLens().getNear(),
                                                       self.quad_camera_node.node().getLens().getFar(),
                                                       shape=(size[1], size[0], 1)))
            if len(observation_spaces) == 1:
                self._observation_space, = observation_spaces
            else:
                self._observation_space = TupleSpace(observation_spaces)
        else:
            self._observation_space = None

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

    def _load_quad(self):
        self.quad_node = self.app.loader.loadModel('iris')
        self.quad_node.reparentTo(self.app.render)

        ambient_color = (.1, .1, .1, 1)
        sun_light_color = (.5, .5, .5, 1)
        self.quad_node.setLightOff()
        ambient_light = self.app.render.attachNewNode(AmbientLight('quad_ambient_light'))
        ambient_light.node().setColor(ambient_color)
        self.quad_node.setLight(ambient_light)
        sun_light = self.app.render.attachNewNode(PointLight('quad_sun_light'))
        sun_light.node().setColor(sun_light_color)
        sun_light.setPos((-2506., -634., 2596.))
        self.quad_node.setLight(sun_light)

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
        hor_car_rot_z = np.array([0, 0, 1])
        hor_car_rot_x = np.cross(hor_car_T[:3, 1], hor_car_rot_z)
        hor_car_rot_y = np.cross(hor_car_rot_z, hor_car_rot_x)
        hor_car_T[:3, :3] = np.array([hor_car_rot_x, hor_car_rot_y, hor_car_rot_z]).T
        return hor_car_T

    def compute_desired_quad_pos_quat(self, offset=None):
        offset = offset if offset is not None else self.offset
        # desired position of the quad is located at offset relative to the car (reoriented so that the car is horizontal)
        hor_car_T = self.hor_car_T
        des_quad_pos = hor_car_T[:3, 3] + hor_car_T[:3, :3].dot(offset)
        # desired rotation of the quad points towards the car while constraining the z-axis to be up
        des_quad_rot_y = hor_car_T[:3, 3] - des_quad_pos
        des_quad_rot_y /= np.linalg.norm(des_quad_rot_y)
        des_quad_rot_z = np.array([0, 0, 1])
        des_quad_rot_x = np.cross(des_quad_rot_y, des_quad_rot_z)
        des_quad_rot_y = np.cross(des_quad_rot_z, des_quad_rot_x)
        des_quad_rot = np.array([des_quad_rot_x, des_quad_rot_y, des_quad_rot_z]).T
        des_quad_quat = tf.quaternion_from_matrix(des_quad_rot)
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
            obs = self.camera_sensor.observe()
            if len(self.sensor_names) == 1:
                obs, = obs
            return obs
        else:
            return None

    def render(self):
        if self._first_render:
            tightness = 1.0
            self._first_render = False
        else:
            tightness = 0.1
        target_node = self.quad_node
        target_T = tf.pose_matrix(target_node.getQuat(), target_node.getPos())
        target_camera_pos = target_T[:3, 3] + target_T[:3, :3].dot(np.array([0., -np.sqrt(3), 1]) * 15)
        self.app.cam.setPos(tuple((1 - tightness) * np.array(self.app.cam.getPos()) + tightness * target_camera_pos))
        self.app.cam.lookAt(target_node)

        # render observation window(s)
        for _ in range(self.quad_camera_sensor.graphics_engine.getNumWindows()):
            self.quad_camera_sensor.graphics_engine.renderFrame()
        self.quad_camera_sensor.graphics_engine.syncFrame()

        # render main window(s)
        for _ in range(self.app.graphicsEngine.getNumWindows()):
            self.app.graphicsEngine.renderFrame()
        self.app.graphicsEngine.syncFrame()
