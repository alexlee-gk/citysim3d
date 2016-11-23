import numpy as np
from panda3d.core import AmbientLight, PointLight
from citysim3d.envs import Panda3dEnv, Panda3dCameraSensor, GeometricCarPanda3dEnv
from citysim3d.spaces import BoxSpace, TranslationAxisAngleSpace
import citysim3d.utils.transformations as tf


class SimpleQuadPanda3dEnv(Panda3dEnv):
    def __init__(self, action_space, sensor_names=None, car_env_class=None,
                 car_action_space=None, car_model_name=None, app=None, dt=None):
        super(SimpleQuadPanda3dEnv, self).__init__(app=app, dt=dt)
        self._action_space = action_space
        self._sensor_names = sensor_names or ['image']
        self.car_env_class = car_env_class or GeometricCarPanda3dEnv
        self.car_action_space = car_action_space or BoxSpace(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        self.car_model_name = car_model_name or ['camaro2']
        if not isinstance(self.car_model_name, (tuple, list)):
            self.car_model_name = [self.car_model_name]
        self.car_env = self.car_env_class(self.car_action_space, sensor_names=[], model_names=self.car_model_name, app=self.app, dt=self.dt)

        assert isinstance(self.action_space, TranslationAxisAngleSpace)

        # modify the car's speed limits so that the car's speed is always a quater of the quad's maximum forward velocity
        self.car_env.speed_offset_space.low[0] = self.action_space.high[1] / 4  # meters per second
        self.car_env.speed_offset_space.high[0] = self.action_space.high[1] / 4
        self.car_node = self.car_env.car_node

        self._load_quad()
        self.prop_angle = 0.0
        self.prop_rpm = 10212

        if self.sensor_names:
            color = depth = False
            for sensor_name in self.sensor_names:
                if sensor_name == 'image':
                    color = True
                elif sensor_name == 'depth_image':
                    depth = True
                else:
                    raise ValueError('Unknown sensor name %s' % sensor_name)
            self.quad_camera_sensor = Panda3dCameraSensor(self.app, color=color, depth=depth)
            self.quad_camera_node = self.quad_camera_sensor.cam
            self.quad_camera_node.reparentTo(self.quad_node)
            self.quad_camera_node.setPos(tuple(np.array([0, -4., 3.]) * -0.02))  # slightly in front of the quad
            self.quad_camera_node.setQuat(tuple(tf.quaternion_about_axis(-np.pi / 3, np.array([1, 0, 0]))))

        self._first_render = True

    @property
    def action_space(self):
        return self._action_space

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

        car_action = self.car_env.action_space.sample()
        self.car_env.step(car_action)

        # update action to be within the action space
        if not self.action_space.contains(action):
            action = self.action_space.clip(action, out=action)

        linear_vel, angular_vel = np.split(action, [3])
        if angular_vel.shape == (1,):
            angular_vel = angular_vel * self.action_space.axis

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
        if self.action_space.axis is not None:
            angular_vel = angular_vel.dot(self.action_space.axis)
        action[:] = np.append(linear_vel, angular_vel)

    def get_state(self):
        quad_T = tf.pose_matrix(self.quad_node.getQuat(), self.quad_node.getPos())
        quad_state = tf.position_axis_angle_from_matrix(quad_T)
        car_state = self.car_env.get_state()
        return np.concatenate([quad_state, car_state])

    def reset(self, state=None):
        self._first_render = True
        if state is None:
            self.car_env.reset(state=None)
            # set the position of the quad to be behind the car
            car_T = tf.pose_matrix(self.car_node.getQuat(), self.car_node.getPos())
            quad_pos = car_T[:3, 3] + car_T[:3, :3].dot(np.array([0., -4., 3.]) * 4)
            # set the rotation of the quad to be the rotation of the car projected so that the z-axis is up
            axis = np.cross(car_T[:3, 2], np.array([0, 0, 1]))
            angle = tf.angle_between_vectors(car_T[:3, 2], np.array([0, 0, 1]))
            if np.isclose(angle, 0.0):
                project_T = np.eye(4)
            else:
                project_T = tf.rotation_matrix(angle, axis)
            quad_T = project_T.dot(car_T)
            quad_T[:3, 3] = quad_pos
        else:
            quad_state, car_state = np.split(state, [6])
            self.car_env.reset(car_state)
            quad_T = tf.position_axis_angle_matrix(quad_state)
        quad_pos = quad_T[:3, 3]
        quad_quat = tf.quaternion_from_matrix(quad_T[:3, :3])
        self.quad_node.setPosQuat(tuple(quad_pos), tuple(quad_quat))

    def observe(self):
        if self.sensor_names:
            return self.quad_camera_sensor.observe()
        else:
            tuple()

    def render(self):
        # update the angle of the propellers (for rendering purposes)
        if self.prop_rpm:
            for quad_prop_id, quad_prop_local_node in enumerate(self.quad_prop_local_nodes):
                is_ccw = quad_prop_id in (1, 2)
                angle = self.prop_angle if is_ccw else -self.prop_angle
                quad_prop_local_node.setQuat(tuple(tf.quaternion_about_axis(angle, np.array([0, 0, 1]))))
        if self._first_render:
            tightness = 1.0
            self._first_render = False
        else:
            tightness = 0.1
        target_node = self.quad_node
        target_T = tf.pose_matrix(target_node.getQuat(), target_node.getPos())
        target_camera_pos = target_T[:3, 3] + target_T[:3, :3].dot(np.array([0., -4., 3.]) * 1)
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
