import numpy as np
import citysim3d.utils.transformations as tf


def get_interaction_matrix_z_forward(x, y, Z, axis=None):
    if axis is None:
        L = np.array([[-1 / Z, 0, x / Z, x * y, -(1 + x ** 2), y],
                      [0, -1 / Z, y / Z, (1 + y ** 2), -x * y, -x]])
    else:
        L = np.array([[-1 / Z, 0, x / Z, np.array([x * y, -(1 + x ** 2), y]).dot(axis)],
                      [0, -1 / Z, y / Z, np.array([(1 + y ** 2), -x * y, -x]).dot(axis)]])
    return L


def get_interaction_matrix_y_forward(x, z, Y, axis=None):
    if axis is None:
        L = np.array([[-1 / Y, x / Y, 0, x * z, z, (1 + x ** 2)],
                      [0, -z / Y, 1 / Y, (1 - z ** 2), -x, -x * z]])
    else:
        L = np.array([[-1 / Y, x / Y, 0, np.array([x * z, z, (1 + x ** 2)]).dot(axis)],
                      [0, -z / Y, 1 / Y, np.array([(1 - z ** 2), -x, -x * z]).dot(axis)]])
    return L


def XYZ_to_xzY(XYZ):
    xzY = np.c_[XYZ[:, 0] / XYZ[:, 1],
                -XYZ[:, 2] / XYZ[:, 1],
                XYZ[:, 1]]
    return xzY


class PointBasedServoingPolicy(object):
    def __init__(self, env, lambda_=1.0, interaction_matrix_type=None, use_car_dynamics=False):
        """
        Args:
            interaction_matrix_type: 'target', 'current' or 'both'. Indicates
                if the interaction matrix should use the depth Z of the target
                points or the current points or a mix of both (where both
                matrices are averages).
        """
        self.env = env
        self.lambda_ = lambda_
        self.interaction_matrix_type = interaction_matrix_type or 'both'
        self.use_car_dynamics = use_car_dynamics

    def act(self, obs):
        points_XYZ = obs['points']
        points_xzY = XYZ_to_xzY(points_XYZ)

        if self.use_car_dynamics:
            # apply zero action to observe changes that are independent from the action. then, restore states.
            random_state = np.random.get_state()
            state = self.env.get_state()
            action = np.zeros(self.env.action_space.shape)
            obs, _, _, _ = self.env.step(action)
            next_points_XYZ = obs['points']
            if next_points_XYZ is None:
                next_points_XYZ = points_XYZ
            next_points_xzY = XYZ_to_xzY(next_points_XYZ)
            s = np.concatenate(next_points_xzY[:, :2])
            self.env.set_state(state)
            np.random.set_state(random_state)
        else:
            s = np.concatenate(points_xzY[:, :2])

        target_points_XYZ = obs['target_points']
        target_points_xzY = XYZ_to_xzY(target_points_XYZ)
        s_target = np.concatenate(target_points_xzY[:, :2])

        L = []
        for point_xzY, target_point_xzY in zip(points_xzY, target_points_xzY):
            if self.interaction_matrix_type == 'target':
                L.append(get_interaction_matrix_y_forward(*point_xzY[:2], Y=target_point_xzY[2]))
            elif self.interaction_matrix_type == 'current':
                L.append(get_interaction_matrix_y_forward(*point_xzY))
            elif self.interaction_matrix_type == 'both':
                L.append(0.5 * get_interaction_matrix_y_forward(*point_xzY[:2], Y=target_point_xzY[2])
                         + 0.5 * get_interaction_matrix_y_forward(*point_xzY))
            else:
                raise ValueError("Invalid interaction matrix option %r" % self.interaction_matrix_type)
        L = np.concatenate(L)

        try:
            action = - self.lambda_ * np.linalg.solve(L.T.dot(L), L.T.dot(s - s_target))
        except np.linalg.linalg.LinAlgError:
            action = np.zeros(self.env.action_space.shape)

        action = self.transform_camera_to_inertial(action)
        return action

    def transform_camera_to_inertial(self, action):
        # transform action from the camera frame to robot's inertial frame
        if not self.env.quad_node.getTransform(self.env.camera_node).isIdentity():
            camera_T = self.env.camera_node.getTransform(self.env.root_node)
            camera_T = np.array(camera_T.getMat()).T

            camera_to_next_camera_T = tf.position_axis_angle_matrix(action)

            next_camera_T = camera_T.dot(camera_to_next_camera_T)

            camera_to_quad_T = self.env.quad_node.getTransform(self.env.camera_node)
            camera_to_quad_T = np.array(camera_to_quad_T.getMat()).T

            next_quad_T = next_camera_T.dot(camera_to_quad_T)

            quad_T = self.env.quad_node.getTransform()
            quad_T = np.array(quad_T.getMat()).T

            quad_to_next_quad_T = tf.inverse_matrix(quad_T).dot(next_quad_T)

            if self.env.action_space.axis is not None:
                # set the rotation of the quad to be the rotation of the car projected so that the z-axis is up
                axis = np.cross(quad_to_next_quad_T[:3, 2], self.env.action_space.axis)
                angle = tf.angle_between_vectors(quad_to_next_quad_T[:3, 2],
                                                 self.env.action_space.axis)
                if np.isclose(angle, 0.0):
                    project_T = np.eye(4)
                else:
                    project_T = tf.rotation_matrix(angle, axis)
                quad_to_next_quad_T = project_T.dot(quad_to_next_quad_T)

            linear_vel, angular_vel = np.split(
                tf.position_axis_angle_from_matrix(quad_to_next_quad_T), [3])
            # project angular_vel onto the axis
            if self.env.action_space.axis is not None:
                angular_vel = angular_vel.dot(self.env.action_space.axis)
            action = np.append(linear_vel, angular_vel)
        return action
