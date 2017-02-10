import numpy as np
from citysim3d.policies import Policy


def get_interaction_matrix_xzY(x, z, Y, axis=None):
    if axis is None:
        L = np.array([[-1 / Y, x / Y, 0, -x * z, -z, (1 + x ** 2)],
                      [0, z / Y, -1 / Y, -(1 + z ** 2), x, x * z]])
    else:
        L = np.array([[-1 / Y, x / Y, 0, np.array([-x * z, -z, (1 + x ** 2)]).dot(axis)],
                      [0, z / Y, -1 / Y, np.array([-(1 + z ** 2), x, x * z]).dot(axis)]])
    return L


def XYZ_to_xzY(XYZ):
    xzY = np.c_[XYZ[:, 0] / XYZ[:, 1],
                XYZ[:, 2] / XYZ[:, 1],
                XYZ[:, 1]]
    return xzY


class ImageBasedServoingPolicy(Policy):
    def __init__(self, env, lambda_=1.0, interaction_matrix_type=None):
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

    def act(self, obs):
        # transform the observed points from the camera's to the inertial's reference frame
        points_XYZ = obs['points']
        if self.env.camera_to_inertial_T is not None:
            points_XYZ = points_XYZ.dot(self.env.camera_to_inertial_T[:3, :3].T) + self.env.camera_to_inertial_T[:3, 3]
        points_xzY = XYZ_to_xzY(points_XYZ)

        s = np.concatenate(points_xzY[:, :2])

        target_points_XYZ = obs['target_points']
        if self.env.camera_to_inertial_T is not None:
            target_points_XYZ = target_points_XYZ.dot(self.env.camera_to_inertial_T[:3, :3].T) + self.env.camera_to_inertial_T[:3, 3]
        target_points_xzY = XYZ_to_xzY(target_points_XYZ)
        s_target = np.concatenate(target_points_xzY[:, :2])

        L = []
        for point_xzY, target_point_xzY in zip(points_xzY, target_points_xzY):
            if self.interaction_matrix_type == 'both':
                L.append(0.5 * get_interaction_matrix_xzY(*point_xzY[:2], Y=target_point_xzY[2], axis=self.env.action_space.axis)
                         + 0.5 * get_interaction_matrix_xzY(*point_xzY, axis=self.env.action_space.axis))
            elif self.interaction_matrix_type == 'target':
                L.append(get_interaction_matrix_xzY(*point_xzY[:2], Y=target_point_xzY[2], axis=self.env.action_space.axis))
            elif self.interaction_matrix_type == 'current':
                L.append(get_interaction_matrix_xzY(*point_xzY, axis=self.env.action_space.axis))
            else:
                raise ValueError("Invalid interaction matrix option %r" % self.interaction_matrix_type)
        L = np.concatenate(L)
        L *= self.env.dt

        try:
            action = - self.lambda_ * np.linalg.solve(L.T.dot(L), L.T.dot(s - s_target))
        except np.linalg.linalg.LinAlgError:
            action = np.zeros(self.env.action_space.shape)
        return action
