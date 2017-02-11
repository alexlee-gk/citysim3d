import numpy as np
from citysim3d.policies import Policy
from citysim3d.utils import transformations as tf
import scipy
import scipy.linalg
import scipy.optimize


def skew(a):
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


class PositionBasedServoingPolicy(Policy):
    def __init__(self, env, lambda_, use_object_frame=True, ignore_rotation=False, use_constrained_opt=True):
        """
        In position-based visual servoing, the translation can be defined as
        the translation of the object in the camera's frame, or as the
        translation of the camera in the target camera's frame. In the first
        scheme, the trajectory in the image of the origin of the object frame
        follows a pure straight line, whereas the camera trajectory does not
        follor a straight line. In the second scheme, the camera trajectory is
        a pure straight line, whereas the image trajectories are less
        satisfactory (some points may even go outside of the field of view).

        F. Chaumette and S. Hutchinson, "Visual servo control. I. Basic
        approaches," in IEEE Robotics & Automation Magazine, vol. 13, no. 4,
        pp. 82-90, Dec. 2006.
        """
        self.env = env
        self.lambda_ = lambda_
        self.use_object_frame = use_object_frame
        self.ignore_rotation = ignore_rotation
        self.use_constrained_opt = use_constrained_opt

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
        if self.use_constrained_opt:
            axis, angle = tf.split_axis_angle(target_to_curr_aa)
            L_aa = np.eye(3) - 0.5 * angle * skew(axis) + (1 - np.sinc(angle) / (np.sinc(0.5 * angle) ** 2)) * (skew(axis) ** 2)
            if self.use_object_frame:
                L = scipy.linalg.block_diag(-np.eye(3), L_aa)
                L[:3, 3:] = skew(curr_to_obj_pos)
                e = np.append(curr_to_obj_pos - target_to_obj_pos, target_to_curr_aa)
            else:
                target_to_curr_rot = tf.matrix_from_axis_angle(target_to_curr_aa)[:3, :3]
                target_to_curr_pos = target_to_obj_pos - target_to_curr_rot.T.dot(curr_to_obj_pos)
                L = scipy.linalg.block_diag(target_to_curr_rot, L_aa)
                e = np.append(target_to_curr_pos, target_to_curr_aa)
            if self.env.action_space.axis is not None:
                L = L.dot(scipy.linalg.block_diag(np.eye(3), self.env.action_space.axis[:, None]))
            else:
                raise NotImplementedError("A least-squares solver with an l2-ball constraint should be used.")
            L *= self.env.dt
            action = -self.lambda_ * scipy.optimize.lsq_linear(L, e, bounds=(self.env.action_space.low, self.env.action_space.high)).x
        else:
            if self.use_object_frame:
                linear_vel = -self.lambda_ * ((target_to_obj_pos - curr_to_obj_pos) + np.cross(curr_to_obj_pos, target_to_curr_aa)) / self.env.dt
                angular_vel = -self.lambda_ * target_to_curr_aa / self.env.dt
            else:
                target_to_curr_rot = tf.matrix_from_axis_angle(target_to_curr_aa)[:3, :3]
                target_to_curr_pos = target_to_obj_pos - target_to_curr_rot.T.dot(curr_to_obj_pos)
                linear_vel = -self.lambda_ * target_to_curr_rot.T.dot(target_to_curr_pos) / self.env.dt
                angular_vel = -self.lambda_ * target_to_curr_aa / self.env.dt
            if self.env.action_space.axis is not None and len(angular_vel) == 3:
                angular_vel = angular_vel.dot(self.env.action_space.axis)
            action = np.r_[linear_vel, angular_vel]
        return action
