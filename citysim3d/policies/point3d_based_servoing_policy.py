import numpy as np
from citysim3d.policies import Policy


def get_interaction_matrix_XYZ(X, Y, Z, axis=None):
    if axis is None:
        L = np.array([[-1, 0, 0, 0, -Z, Y],
                      [0, -1, 0, Z, 0, -X],
                      [0, 0, -1, -Y, X, 0]])
    else:
        L = np.array([[-1, 0, 0, np.array([0, -Z, Y]).dot(axis)],
                      [0, -1, 0, np.array([Z, 0, -X]).dot(axis)],
                      [0, 0, -1, np.array([-Y, X, 0]).dot(axis)]])
    return L


class Point3dBasedServoingPolicy(Policy):
    def __init__(self, env, lambda_=1.0):
        self.env = env
        self.lambda_ = lambda_

    def act(self, obs):
        # transform the observed points from the camera's to the inertial's reference frame
        points_XYZ = obs['points']
        if self.env.camera_to_inertial_T is not None:
            points_XYZ = points_XYZ.dot(self.env.camera_to_inertial_T[:3, :3].T) + self.env.camera_to_inertial_T[:3, 3]
        s = np.concatenate(points_XYZ)

        target_points_XYZ = obs['target_points']
        if self.env.camera_to_inertial_T is not None:
            target_points_XYZ = target_points_XYZ.dot(self.env.camera_to_inertial_T[:3, :3].T) + self.env.camera_to_inertial_T[:3, 3]
        s_target = np.concatenate(target_points_XYZ)

        L = []
        for point_XYZ, target_point_XYZ in zip(points_XYZ, target_points_XYZ):
            L.append(get_interaction_matrix_XYZ(*point_XYZ, axis=self.env.action_space.axis))
        L = np.concatenate(L)
        L *= self.env.dt

        try:
            action = - self.lambda_ * np.linalg.solve(L.T.dot(L), L.T.dot(s - s_target))
        except np.linalg.linalg.LinAlgError:
            action = np.zeros(self.env.action_space.shape)
        return action
