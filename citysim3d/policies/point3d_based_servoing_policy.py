import numpy as np
from citysim3d.policies import PointBasedServoingPolicy


def get_interaction_matrix_XYZ(X, Y, Z):
    L = np.array([[-1, 0, 0, 0, -Z, Y],
                  [0, -1, 0, Z, 0, -X],
                  [0, 0, -1, -Y, X, 0]])
    return L


class Point3dBasedServoingPolicy(PointBasedServoingPolicy):
    def __init__(self, env, lambda_=1.0, use_car_dynamics=False):
        self.env = env
        self.lambda_ = lambda_
        self.use_car_dynamics = use_car_dynamics

    def act(self, obs):
        points_XYZ = obs[0]
        target_points_XYZ = obs[len(obs) // 2]

        s_target = np.concatenate(target_points_XYZ)

        if self.use_car_dynamics:
            # apply zero action to observe changes that are independent from the action. then, restore states.
            random_state = np.random.get_state()
            state = self.env.get_state()
            action = np.zeros(self.env.action_space.shape)
            obs, _, _, _ = self.env.step(action)
            next_points_XYZ = obs[0]
            if next_points_XYZ is None:
                next_points_XYZ = points_XYZ
            s = np.concatenate(next_points_XYZ)
            self.env.set_state(state)
            np.random.set_state(random_state)
        else:
            s = np.concatenate(points_XYZ)

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
