import numpy as np
import os.path
from collections import defaultdict
import collada
from panda3d.core import AmbientLight, PointLight
from citysim3d.envs import Panda3dEnv, Panda3dCameraSensor
from citysim3d.spaces import BoxSpace, DictSpace
import citysim3d.utils.transformations as tf


class CarPanda3dEnv(Panda3dEnv):
    def __init__(self, action_space, sensor_names=None, model_names=None, app=None, dt=None):
        super(CarPanda3dEnv, self).__init__(app=app, dt=dt)
        self._action_space = action_space
        self._sensor_names = sensor_names if sensor_names is not None else ['image']  # don't override empty list
        self.model_names = model_names or ['camaro2']
        if not isinstance(self.model_names, (tuple, list)):
            raise ValueError('model_names should be a tuple or a list, but %r was given.' % model_names)

        # state
        self._speed = 1.0
        self._lane_offset = 2.0
        self._straight_dist = 0.0
        self._model_name = None
        # road properties
        self._lane_width = 4.0
        self._num_lanes = 2

        self.speed_offset_space = BoxSpace([0.0, 0.5 * self._lane_width],
                                           [10.0, (self._num_lanes - 0.5) * self._lane_width])

        self._load_city()
        self.car_node = self.app.render.attachNewNode('car')
        self._car_local_node = None
        self._car_local_nodes = dict()

        self.model_name = self.model_names[0]  # first car by default

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
            self.car_camera_sensor = self.camera_sensor = Panda3dCameraSensor(self.app, color=color, depth=depth)
            self.car_camera_node = self.camera_node = self.car_camera_sensor.cam
            self.car_camera_node.reparentTo(self.car_node)
            self.car_camera_node.setPos(tuple(np.array([0, 1, 2])))  # slightly in front of the car
            self.car_camera_node.setName('car_camera')

            for sensor_name in self.sensor_names:
                if sensor_name == 'image':
                    observation_spaces[sensor_name] = BoxSpace(0, 255, shape=(480, 640, 3), dtype=np.uint8)
                elif sensor_name == 'depth_image':
                    observation_spaces[sensor_name] = BoxSpace(self.car_camera_node.node().getLens().getNear(),
                                                               self.car_camera_node.node().getLens().getFar(),
                                                               shape=(480, 640, 1))
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

    def _load_city(self):
        try:
            self.skybox_node = self.app.loader.loadModel("skyboxes/01-clean-day/skybox-mesh.egg")
            self.skybox_node.reparentTo(self.app.render)
        except IOError:
            print("You probably only have the public models. Skipping loading file skyboxes/01-clean-day/skybox-mesh.egg")

        try:
            self.city_node = self.app.loader.loadModel("levels/urban-level-02-medium.egg")
            self.city_node.reparentTo(self.app.render)

            self.ambient_light = self.app.render.attachNewNode(AmbientLight('ambient_light'))
            self.ambient_light.node().setColor((1, 1, 1, 1))
            self.city_node.setLight(self.ambient_light)

            self.sun_light = self.app.render.attachNewNode(PointLight('sun_light'))
            self.sun_light.node().setColor((.2, .2, .2, 1))
            self.sun_light.setPos((-2506., -634., 2596.))
            self.city_node.setLight(self.sun_light)
        except IOError:
            print("You probably only have the public models. Skipping loading file levels/urban-level-02-medium.egg")

    def _load_and_get_car_local(self, model_name):
        car_local_node = self.app.loader.loadModel(model_name)
        # translate model so that the bottom of it is at a height of 0 in the local reference frame
        car_local_pos = car_local_node.getPos()
        car_local_pos[2] = -car_local_node.getTightBounds()[0][2]
        car_local_node.setPos(tuple(car_local_pos))

        # cars need lights that are different from the scene lights
        if model_name in ('camaro2', 'sport'):
            ambient_color = (.1, .1, .1, 1)
            sun_light_color = (.8, .8, .8, 1)
        elif model_name == 'mazda6':
            ambient_color = (.1, .1, .1, 1)
            sun_light_color = (1, 1, 1, 1)
        elif model_name == 'mitsubishi_lancer_evo':
            ambient_color = (.2, .2, .2, 1)
            sun_light_color = (1, 1, 1, 1)
        else:
            ambient_color = (.3, .3, .3, 1)
            sun_light_color = (1, 1, 1, 1)
        car_local_node.setLightOff()
        ambient_light = self.app.render.attachNewNode(AmbientLight('car_ambient_light'))
        ambient_light.node().setColor(ambient_color)
        car_local_node.setLight(ambient_light)
        sun_light = self.app.render.attachNewNode(PointLight('car_sun_light'))
        sun_light.node().setColor(sun_light_color)
        sun_light.setPos((-2506., -634., 2596.))
        car_local_node.setLight(sun_light)
        return car_local_node

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        if model_name != self._model_name:
            self._model_name = model_name
            if self._car_local_node is not None:
                self._car_local_node.detachNode()
            self._car_local_node = self._car_local_nodes.get(self.model_name)
            if self._car_local_node is None:
                self._car_local_node = self._load_and_get_car_local(self.model_name)
                self._car_local_nodes[self.model_name] = self._car_local_node
            self._car_local_node.reparentTo(self.car_node)

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        self._speed = np.clip(speed, self.speed_offset_space.low[0], self.speed_offset_space.high[0])

    @property
    def lane_offset(self):
        return self._lane_offset

    @lane_offset.setter
    def lane_offset(self, lane_offset):
        self._lane_offset = np.clip(lane_offset, self.speed_offset_space.low[1], self.speed_offset_space.high[1])

    @property
    def straight_dist(self):
        return self._straight_dist

    @straight_dist.setter
    def straight_dist(self, straight_dist):
        self._straight_dist = straight_dist

    def observe(self):
        if self.sensor_names:
            return dict(zip(self.sensor_names, self.camera_sensor.observe()))
        else:
            return dict()

    def render(self):
        if self._first_render:
            tightness = 1.0
            self._first_render = False
        else:
            tightness = 0.1
        target_node = self.car_node
        target_T = tf.pose_matrix(target_node.getQuat(), target_node.getPos())
        target_camera_pos = target_T[:3, 3] + target_T[:3, :3].dot(np.array([0., -4., 3.]) * 4)
        self.app.cam.setPos(tuple((1 - tightness) * np.array(self.app.cam.getPos()) + tightness * target_camera_pos))
        self.app.cam.lookAt(target_node)

        # render observation window(s)
        for _ in range(self.car_camera_sensor.graphics_engine.getNumWindows()):
            self.car_camera_sensor.graphics_engine.renderFrame()
        self.car_camera_sensor.graphics_engine.syncFrame()

        # render main window(s)
        for _ in range(self.app.graphicsEngine.getNumWindows()):
            self.app.graphicsEngine.renderFrame()
        self.app.graphicsEngine.syncFrame()


class StraightCarPanda3dEnv(CarPanda3dEnv):
    def __init__(self, action_space, sensor_names=None, model_names=None, app=None, dt=None):
        super(StraightCarPanda3dEnv, self).__init__(action_space, sensor_names=sensor_names, model_names=model_names, app=app, dt=dt)
        self.dist_space = BoxSpace(0, 275 + 225)
        # minimum and maximum position of the car
        # [-51 - 6, -275, 10.7]
        # [-51 + 6, 225, 10.7]

    @property
    def straight_dist(self):
        return self._straight_dist

    @straight_dist.setter
    def straight_dist(self, straight_dist):
        self._straight_dist = np.clip(straight_dist, self.dist_space.low, self.dist_space.high)

    @property
    def position(self):
        return np.array([-51 + self.lane_offset, -275 + self.straight_dist, 10.7])

    def step(self, action):
        forward_acceleration, lateral_velocity = action
        self.speed += forward_acceleration * self.dt
        self.lane_offset += lateral_velocity * self.dt
        self.straight_dist += self.speed * self.dt
        self.car_node.setPos(tuple(self.position))
        return self.observe(), None, False, dict()

    def reset(self, state=None):
        self._first_render = True
        if state is None:
            speed, lane_offset = self.speed_offset_space.sample()
            straight_dist = self.dist_space.sample()
            model_name_ind = np.random.randint(0, len(self.model_names))
            state = speed, lane_offset, straight_dist, model_name_ind
        self.set_state(state)
        return self.observe()

    def get_state(self):
        model_name_ind = self.model_names.index(self.model_name)
        return np.array([self.speed, self.lane_offset, self.straight_dist, model_name_ind])

    def set_state(self, state):
        speed, lane_offset, straight_dist, model_name_ind = state
        model_name_ind = int(model_name_ind)
        self.speed, self.lane_offset, self.straight_dist = speed, lane_offset, straight_dist
        self.model_name = self.model_names[model_name_ind]
        self.car_node.setPos(tuple(self.position))


class SimpleGeometricCarPanda3dEnv(CarPanda3dEnv):
    def __init__(self, action_space, sensor_names=None, model_names=None, app=None, dt=None):
        super(SimpleGeometricCarPanda3dEnv, self).__init__(action_space, sensor_names=sensor_names, model_names=model_names, app=app, dt=dt)
        self._graph_collada_fname = os.path.expandvars('${CITYSIM3D_DIR}/models/megacity-urban-construction-kit/levels/'
                                                       'urban-level-02-medium-road-directed-graph.dae')
        self._faces_collada_fname = os.path.expandvars('${CITYSIM3D_DIR}/models/megacity-urban-construction-kit/levels/'
                                                       'urban-level-02-medium-road-faces.dae')
        self._faces_collada_file = None
        self._edge_normals = {}
        self._last_edge = self._edge_normal = None
        self._build_graph()
        # for sampling turns
        self.angle_variance = np.pi / 2.0
        self.angle_thresh = np.pi * 3.0 / 4.0
        # state
        self._start_ind, self._end_ind = self.sample_vertex_inds()
        self.car_node.setPosQuat(*self.pos_quat)

    @property
    def straight_dist(self):
        return self._straight_dist

    @straight_dist.setter
    def straight_dist(self, straight_dist):
        self._straight_dist = np.clip(straight_dist, 0.0, self.max_straight_dist)

    @property
    def start_pos(self):
        return self._points[self._start_ind]

    @property
    def end_pos(self):
        return self._points[self._end_ind]

    @property
    def edge_normal(self):
        if (self._start_ind, self._end_ind) != self._last_edge:
            self._last_edge = (self._start_ind, self._end_ind)
            self._edge_normal = self._get_or_compute_edge_normal((self._start_ind, self._end_ind))
        return self._edge_normal

    @property
    def max_straight_dist(self):
        return np.linalg.norm(self.end_pos - self.start_pos)

    def _build_graph(self):
        col = collada.Collada(self._graph_collada_fname)
        geom = col.geometries[0]
        lineset = geom.primitives[0]
        self._graph = defaultdict(set)
        for node1, node2 in lineset.vertex_index:
            self._graph[node1].add(node2)
        self._points = lineset.vertex

    def _get_or_compute_edge_normal(self, edge_inds):
        edge_normal = self._edge_normals.get(edge_inds)
        if edge_normal is None:
            if self._faces_collada_file is None:
                self._faces_collada_file = collada.Collada(self._faces_collada_fname)
            geom = self._faces_collada_file.geometries[0]
            triset = geom.primitives[0]
            start_ind, end_ind = edge_inds
            start_point, end_point = self._points[start_ind], self._points[end_ind]
            # alternative implementation that loops over triangles and is ~100x slower
            # # iterate over triangles and check if the start and end points belong to each triangle
            # for tri_inds in triset.vertex_index:
            #     start_tri_ind = -1
            #     end_tri_ind = -1
            #     tri_points = triset.vertex[tri_inds]
            #     for tri_ind, tri_point in enumerate(tri_points):
            #         if np.allclose(tri_point, start_point):
            #             start_tri_ind = tri_ind
            #         if np.allclose(tri_point, end_point):
            #             end_tri_ind = tri_ind
            #     if start_tri_ind == -1 or end_tri_ind == -1:
            #         continue
            #     # check if start ind comes after end ind in the triangle
            #     if (start_tri_ind % 3) == ((end_tri_ind + 1) % 3):
            #         edge_normal = np.cross(tri_points[1] - tri_points[0], tri_points[2] - tri_points[0])
            #         edge_normal /= np.linalg.norm(edge_normal)
            #         break
            start_tri_inds = np.where(np.isclose(triset.vertex[triset.vertex_index.flatten()], start_point).all(axis=1))[0]
            end_tri_inds = np.where(np.isclose(triset.vertex[triset.vertex_index.flatten()], end_point).all(axis=1))[0]
            for start_tri_ind in start_tri_inds:
                for end_tri_ind in end_tri_inds:
                    same_triangle = start_tri_ind // 3 == end_tri_ind // 3
                    if same_triangle and (start_tri_ind % 3) == ((end_tri_ind + 1) % 3):
                        tri_inds = triset.vertex_index[start_tri_ind // 3]
                        tri_points = triset.vertex[tri_inds]
                        edge_normal = np.cross(tri_points[1] - tri_points[0], tri_points[2] - tri_points[0])
                        edge_normal /= np.linalg.norm(edge_normal)
                        break
            self._edge_normals[edge_inds] = edge_normal
        return edge_normal

    def _next_ind(self, ind0, ind1):
        dir = self._points[ind1] - self._points[ind0]
        next_inds = list(self._graph[ind1])
        if len(next_inds) == 1:
            next_ind, = next_inds
        else:
            angle_changes = []
            for next_ind in next_inds:
                next_dir = self._points[next_ind] - self._points[ind1]
                angle_changes.append(tf.angle_between_vectors(dir, next_dir))
            angle_changes = np.asarray(angle_changes)
            probs = np.exp(-(angle_changes ** 2) / (2. * self.angle_variance))
            probs[np.abs(angle_changes) > self.angle_thresh] = 0.0
            probs /= probs.sum()
            next_ind = np.random.choice(next_inds, p=probs)
        return next_ind

    def _compute_rotation(self, rot_y, rot_z=None):
        rot_y /= np.linalg.norm(rot_y)
        if rot_z is None:
            up_v = np.array([0., 0., 1.])
            rot_x = np.cross(rot_y, up_v)
            rot_z = np.cross(rot_x, rot_y)
        else:
            rot_z /= np.linalg.norm(rot_z)
            rot_x = np.cross(rot_y, rot_z)
        rot = np.array([rot_x, rot_y, rot_z]).T
        return rot

    @property
    def transform(self):
        start_T = np.eye(4)
        start_T[:3, :3] = self._compute_rotation(self.end_pos - self.start_pos, self.edge_normal)
        start_T[:3, 3] = self.start_pos
        translate_to_lane_T = tf.translation_matrix(np.array([self._lane_offset, self._straight_dist, 0.]))
        T = start_T.dot(translate_to_lane_T)
        return T

    @property
    def pos_quat(self):
        car_T = self.transform
        return tuple(car_T[:3, 3]), tuple(tf.quaternion_from_matrix(car_T[:3, :3]))

    def step(self, action):
        forward_acceleration, lateral_velocity = action
        self.speed += forward_acceleration * self.dt
        self.lane_offset += lateral_velocity * self.dt

        delta_dist = self.speed * self.dt
        while delta_dist > 0.0:
            remaining_dist = self.max_straight_dist - self._straight_dist
            if delta_dist < remaining_dist:
                self._straight_dist += delta_dist
                delta_dist = 0.0
            else:
                delta_dist -= remaining_dist
                self._straight_dist = 0.0
                # advance to next segment
                self._start_ind, self._end_ind = \
                    self._end_ind, self._next_ind(self._start_ind, self._end_ind)
        self.car_node.setPosQuat(*self.pos_quat)
        return self.observe(), None, False, dict()

    def reset(self, state=None):
        self._first_render = True
        if state is None:
            speed, lane_offset = self.speed_offset_space.sample()
            self._start_ind, self._end_ind = self.sample_vertex_inds()
            straight_dist = np.random.uniform(0.0, self.max_straight_dist)
            model_name_ind = np.random.randint(0, len(self.model_names))
            state = speed, lane_offset, straight_dist, self._start_ind, self._end_ind, model_name_ind
        self.set_state(state)
        return self.observe()

    def get_state(self):
        model_name_ind = self.model_names.index(self.model_name)
        return np.array([self.speed, self.lane_offset, self.straight_dist, self._start_ind, self._end_ind, model_name_ind])

    def set_state(self, state):
        speed, lane_offset, straight_dist, start_ind, end_ind, model_name_ind = state
        self._start_ind, self._end_ind = int(start_ind), int(end_ind)
        model_name_ind = int(model_name_ind)
        self.speed, self.lane_offset, self.straight_dist = speed, lane_offset, straight_dist
        self.model_name = self.model_names[model_name_ind]
        self.car_node.setPosQuat(*self.pos_quat)

    def sample_vertex_inds(self, start_ind=None):
        start_ind = start_ind or np.random.choice(list(self._graph.keys()))
        end_ind = np.random.choice(list(self._graph[start_ind]))
        return start_ind, end_ind


class GeometricCarPanda3dEnv(SimpleGeometricCarPanda3dEnv):
    def __init__(self, action_space, sensor_names=None, model_names=None, app=None, dt=None):
        CarPanda3dEnv.__init__(self, action_space, sensor_names=sensor_names, model_names=model_names, app=app, dt=dt)
        self._graph_collada_fname = os.path.expandvars('${CITYSIM3D_DIR}/models/megacity-urban-construction-kit/levels/'
                                                       'urban-level-02-medium-road-directed-graph.dae')
        self._faces_collada_fname = os.path.expandvars('${CITYSIM3D_DIR}/models/megacity-urban-construction-kit/levels/'
                                                       'urban-level-02-medium-road-faces.dae')
        self._faces_collada_file = None
        self._edge_normals = {}
        self._last_edge = self._edge_normal = None
        self._build_graph()
        # for sampling turns
        self.angle_variance = np.pi / 2.0
        self.angle_thresh = np.pi * 3.0 / 4.0
        # state
        self._turn_angle = None  # angle along current curve (defined by two adjacent edges)
        self._start_ind, self._middle_ind, self._end_ind = self.sample_vertex_inds()  # SimpleCarPanda3dEnv's start_ind and end_ind are CarPanda3dEnv's start_ind and middle_ind
        self.car_node.setPosQuat(*self.pos_quat)

    @property
    def middle_pos(self):
        return self._points[self._middle_ind]

    @property
    def start_rot(self):
        pt0 = self._points[self._start_ind]
        pt1 = self._points[self._middle_ind]
        return self._compute_rotation(pt1 - pt0, self._get_or_compute_edge_normal((self._start_ind, self._middle_ind)))

    @property
    def middle_rot(self):
        pt1 = self._points[self._middle_ind]
        pt2 = self._points[self._end_ind]
        return self._compute_rotation(pt2 - pt1, self._get_or_compute_edge_normal((self._middle_ind, self._end_ind)))

    @property
    def end_rot(self):
        pt1 = self._points[self._middle_ind]
        pt2 = self._points[self._end_ind]
        return self._compute_rotation(pt2 - pt1, self._get_or_compute_edge_normal((self._middle_ind, self._end_ind)))

    @property
    def start_T(self):
        start_T = np.eye(4)
        start_T[:3, :3] = self.start_rot
        start_T[:3, 3] = self.start_pos
        return start_T

    @property
    def middle_T(self):
        middle_T = np.eye(4)
        middle_T[:3, :3] = self.middle_rot
        middle_T[:3, 3] = self.middle_pos
        return middle_T

    @property
    def start_local_pos(self):
        # start_local_pos = tf.inverse_matrix(self.start_T).dot(np.r_[self.start_pos, 1])[:3]
        # assert np.allclose(start_local_pos, 0.0)
        return np.zeros(2)

    @property
    def middle_local_pos(self):
        # middle_local_pos = tf.inverse_matrix(self.start_T).dot(np.r_[self.middle_pos, 1])[:3]
        # assert np.allclose(middle_local_pos[[0, 2]], 0.0)
        return np.array([0, np.linalg.norm(self.middle_pos - self.start_pos)])

    @property
    def project_T(self):
        """
        Transforms the second plane (defined by normal self.middle_rot[:, 2])
        so that it is parallel to the first plane (defined by normal
        self.start_rot[:, 2]).
        """
        axis = np.cross(self.middle_rot[:, 2], self.start_rot[:, 2])
        angle = tf.angle_between_vectors(self.middle_rot[:, 2], self.start_rot[:, 2])
        if np.isclose(angle, 0.0):
            project_T = np.eye(4)
        else:
            project_T = tf.rotation_matrix(angle, axis, point=self.middle_pos)
        # assert np.allclose(project_T.dot(self.middle_T)[:3, 3], self.middle_pos)
        # assert np.allclose(project_T.dot(self.middle_T)[:3, 2], self.start_rot[:, 2], atol=1e-7)
        return project_T

    @property
    def end_local_pos(self):
        end_local_pos = tf.inverse_matrix(self.start_T).dot(self.project_T.dot(np.r_[self.end_pos, 1]))
        # assert np.allclose(end_local_pos[2], 0.0, atol=1e-5)
        return end_local_pos[:2]

    @property
    def max_straight_dist(self):
        # assert np.isclose(self.middle_local_pos[1], np.linalg.norm(self.middle_pos - self.start_pos))
        return self.middle_local_pos[1]

    @property
    def max_turn_angle(self):
        angle = tf.angle_between_vectors(self.middle_local_pos - self.start_local_pos, self.end_local_pos - self.middle_local_pos)
        assert 0 <= angle <= np.pi
        return angle

    @property
    def turn_radius(self):
        return self.dist_to_center + self.left_turn * self._lane_offset

    @property
    def left_turn(self):
        return np.sign(np.cross(self.middle_local_pos - self.start_local_pos, self.end_local_pos - self.middle_local_pos))

    @property
    def turn_dist_offset(self):
        """
        Distance from the end of the current edge where the curve starts, which
        is the same the distance from the start of the next edge where the
        curve ends.
        """
        if np.isclose(self.max_turn_angle, np.pi):  # U-turn
            turn_dist_offset = 0.0
        else:
            turn_dist_offset = (self.dist_to_center) / np.tan((np.pi - self.max_turn_angle) / 2)
        return turn_dist_offset

    @property
    def dist_to_center(self):
        """
        Perpendicular distance from lane origin to center of turning. The
        minimum is chosen so that turn_dist_offset <= max_straight_dist
        """
        min_dist_to_center = self.max_straight_dist * np.tan((np.pi - self.max_turn_angle) / 2)
        return min(self._num_lanes * self._lane_width, min_dist_to_center)

    @property
    def transform(self):
        assert (self._straight_dist is None) != (self._turn_angle is None)
        if self._straight_dist is not None:
            translate_to_lane_T = tf.translation_matrix(np.array([self._lane_offset, self._straight_dist, 0.]))
            T = self.start_T.dot(translate_to_lane_T)
        else:  # self._turn_angle is not None
            middle_T = self.middle_T
            left_turn = self.left_turn
            translate_to_center_T = tf.translation_matrix(
                np.array([-left_turn * self.dist_to_center,
                          self.turn_dist_offset,
                          0.]))
            rotate_about_center_T = tf.rotation_matrix(
                left_turn * (self._turn_angle - self.max_turn_angle), middle_T[:3, 2])
            translate_to_lane_T = tf.translation_matrix(
                np.array([left_turn * self.dist_to_center + self._lane_offset, 0., 0.]))
            T = middle_T.dot(translate_to_center_T.dot(rotate_about_center_T.dot(translate_to_lane_T)))
            if self._turn_angle < self.max_turn_angle / 2:
                T = self.project_T.dot(T)

        # distance from the next edge
        if self._straight_dist is not None:
            dist = self.max_straight_dist - self._straight_dist - self.turn_dist_offset + (self.max_turn_angle / 2) * self.turn_radius
        else:
            if self._turn_angle < self.max_turn_angle / 2:
                dist = (self.max_turn_angle / 2 - self._turn_angle) * self.turn_radius
            else:
                dist = None  # hard to compute
        dist_thresh = 5.0  # start transition when the next edge is closer than the threshold
        if dist is not None and dist < dist_thresh:
            # start with transform aligned with start_rot but end with transform aligned with middle_rot
            fraction = dist / dist_thresh
            axis = np.cross(self.middle_rot[:, 2], self.start_rot[:, 2])
            angle = - (1 - fraction) * tf.angle_between_vectors(self.middle_rot[:, 2], self.start_rot[:, 2])
            if not np.isclose(angle, 0.0):
                project_T = tf.rotation_matrix(angle, axis, point=self.middle_pos)
                T = project_T.dot(T)
        return T

    def step(self, action):
        forward_acceleration, lateral_velocity = action
        self.speed += forward_acceleration * self.dt
        self.lane_offset += lateral_velocity * self.dt

        delta_dist = self.speed * self.dt
        while delta_dist > 0.0:
            assert (self._straight_dist is None) != (self._turn_angle is None)
            if self._straight_dist is not None:
                remaining_dist = self.max_straight_dist - self._straight_dist - self.turn_dist_offset
                if delta_dist < remaining_dist:
                    self._straight_dist += delta_dist
                    delta_dist = 0.0
                else:
                    delta_dist -= remaining_dist
                    self._straight_dist = None
                    self._turn_angle = 0.0
            else:  # self._turn_angle is not None
                remaining_dist = (self.max_turn_angle - self._turn_angle) * self.turn_radius
                if delta_dist < remaining_dist:
                    self._turn_angle += delta_dist / self.turn_radius
                    delta_dist = 0.0
                else:
                    delta_dist -= remaining_dist
                    self._turn_angle = None
                    self._straight_dist = self.turn_dist_offset
                    # advance to next segment
                    self._start_ind = self._middle_ind
                    self._middle_ind = self._end_ind
                    self._end_ind = self._next_ind(self._start_ind, self._middle_ind)
        self.car_node.setPosQuat(*self.pos_quat)
        return self.observe(), None, False, dict()

    def reset(self, state=None):
        self._first_render = True
        if state is None:
            speed, lane_offset = self.speed_offset_space.sample()
            self._start_ind, self._middle_ind, self._end_ind = self.sample_vertex_inds()
            straight_dist = np.random.uniform(0.0, self.max_straight_dist)  # distance along current edge
            turn_angle = None  # angle along current curve (defined by two adjacent edges)
            model_name_ind = np.random.randint(0, len(self.model_names))
            state = speed, lane_offset, straight_dist, turn_angle, self._start_ind, self._middle_ind, self._end_ind, model_name_ind
        self.set_state(state)
        return self.observe()

    def get_state(self):
        # convert None to -1 so that the state is a numeric array (as opposed to object array)
        straight_dist = self._straight_dist if self._straight_dist is not None else -1
        turn_angle = self._turn_angle if self._turn_angle is not None else -1
        model_name_ind = self.model_names.index(self.model_name)
        return np.array([self.speed, self.lane_offset, straight_dist, turn_angle,
                         self._start_ind, self._middle_ind, self._end_ind, model_name_ind])

    def set_state(self, state):
        speed, lane_offset, straight_dist, turn_angle, start_ind, middle_ind, end_ind, model_name_ind = state
        self._start_ind, self._middle_ind, self._end_ind = int(start_ind), int(middle_ind), int(end_ind)
        model_name_ind = int(model_name_ind)
        # convert -1 to None
        if straight_dist == -1:
            straight_dist = None
        if turn_angle == -1:
            turn_angle = None
        self.speed = speed
        self.lane_offset = lane_offset
        self._straight_dist = straight_dist  # set self._straight_dist directly to prevent clipping of None
        self._turn_angle = turn_angle
        self.model_name = self.model_names[model_name_ind]
        self.car_node.setPosQuat(*self.pos_quat)

    def sample_vertex_inds(self, start_ind=None, middle_ind=None):
        if start_ind is None:
            start_ind = np.random.choice(list(self._graph.keys()))
        if middle_ind is None:
            middle_ind = np.random.choice(list(self._graph[start_ind]))
        elif middle_ind not in self._graph[start_ind]:
            raise ValueError("Invalid start_ind %d and end_ind %d" % (start_ind, middle_ind))
        end_ind = self._next_ind(start_ind, middle_ind)
        return start_ind, middle_ind, end_ind
