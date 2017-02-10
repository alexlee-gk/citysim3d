import numpy as np
from citysim3d.policies import Policy
from citysim3d.utils import transformations as tf


class PositionBasedServoingPolicy(Policy):
    def __init__(self, env, lambda_, straight_trajectory=True, ignore_rotation=False):
        self.env = env
        self.lambda_ = lambda_
        self.straight_trajectory = straight_trajectory
        self.ignore_rotation = ignore_rotation

    def act(self, obs):
        target_to_obj_T = obs['target_transform']
        curr_to_obj_T = obs['transform']
        target_to_obj_pos = target_to_obj_T[:3, 3]
        curr_to_obj_pos = curr_to_obj_T[:3, 3]
        if self.ignore_rotation:
            target_to_curr_aa = np.zeros(3)
        else:
            # more stable version of this
            # target_to_curr_T = target_to_obj_T.dot(tf.inverse_matrix(curr_to_obj_T))
            target_to_obj_quat = tf.quaternion_from_matrix(target_to_obj_T)
            curr_to_obj_quat = tf.quaternion_from_matrix(curr_to_obj_T)
            target_to_curr_quat = tf.quaternion_multiply(target_to_obj_quat,
                                                         tf.quaternion_inverse(curr_to_obj_quat))
            target_to_curr_aa = tf.axis_angle_from_quaternion(target_to_curr_quat)
            if self.env.action_space.axis is not None:
                target_to_curr_aa = target_to_curr_aa.dot(self.env.action_space.axis) * self.env.action_space.axis
        if self.straight_trajectory:
            target_to_curr_rot = tf.matrix_from_axis_angle(target_to_curr_aa)[:3, :3]
            target_to_curr_pos = target_to_obj_pos - target_to_curr_rot.T.dot(curr_to_obj_pos)
            linear_vel = -self.lambda_ * target_to_curr_rot.T.dot(target_to_curr_pos) / self.env.dt
            angular_vel = -self.lambda_ * target_to_curr_aa / self.env.dt
        else:
            linear_vel = -self.lambda_ * ((target_to_obj_pos - curr_to_obj_pos) + np.cross(curr_to_obj_pos, target_to_curr_aa)) / self.env.dt
            angular_vel = -self.lambda_ * target_to_curr_aa / self.env.dt
        if self.env.action_space.axis is not None:
            angular_vel = angular_vel.dot(self.env.action_space.axis)
        action = np.r_[linear_vel, angular_vel]
        return action
