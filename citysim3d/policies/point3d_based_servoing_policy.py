import numpy as np
from citysim3d.policies import PointBasedServoingPolicy


def get_interaction_matrix_XYZ(X, Y, Z):
    L = np.array([[-1, 0, 0, 0, -Z, Y],
                  [0, -1, 0, Z, 0, -X],
                  [0, 0, -1, -Y, X, 0]])
    return L


class Point3dBasedServoingPolicy(PointBasedServoingPolicy):
    def __init__(self, env, lambda_=1.0):
        self.env = env
        self.lambda_ = lambda_

    def act(self, obs):
        points_XYZ = obs[0]
        target_points_XYZ = obs[len(obs) // 2]

        s = np.concatenate(points_XYZ)
        s_target = np.concatenate(target_points_XYZ)

        L = []
        for point_XYZ, target_point_XYZ in zip(points_XYZ, target_points_XYZ):
            L.append(get_interaction_matrix_XYZ(*point_XYZ))
        L = np.concatenate(L)

        try:
            action = - self.lambda_ * np.linalg.solve(L.T.dot(L), L.T.dot(s - s_target))
        except np.linalg.linalg.LinAlgError:
            action = np.zeros(self.env.action_space.shape)

        action = self.transform_camera_to_inertial(action)
        return action
