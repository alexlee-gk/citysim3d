import os
import argparse
import numpy as np
import pickle
import cv2
from panda3d.core import loadPrcFile
from citysim3d.spaces import TranslationAxisAngleSpace
from citysim3d.envs import BboxSimpleQuadPanda3dEnv, Bbox3dSimpleQuadPanda3dEnv, ServoingDesignedFeaturesSimpleQuadPanda3dEnv
from citysim3d.envs import ServoingEnv, NormalizedEnv
from citysim3d.policies import PointBasedServoingPolicy, Point3dBasedServoingPolicy
import citysim3d.utils.panda3d_util as putil


assert "CITYSIM3D_DIR" in os.environ
loadPrcFile(os.path.expandvars('${CITYSIM3D_DIR}/config.prc'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, choices=('bbox', 'bbox3d', 'designed'))
    parser.add_argument('--feature_type', type=str, choices=('sift', 'surf', 'orb'), help='[default=sift] (only valid for env=designed)')
    parser.add_argument('--filter_features', type=int, help='[default=1] whether to filter out key points that are not on the object in the current image (only valid for env=designed)')
    parser.add_argument('--use_3d_pol', action='store_true', help='use policy that minimizes the error of 3d points (as opposed to projected 2d points)')
    parser.add_argument('--use_car_dynamics', '--use_car_dyn', action='store_true')
    parser.add_argument('--lambda_', '--lambda', type=float, default=1.0)
    parser.add_argument('--interaction_matrix_type', '--inter_mat_type', type=str, choices=('target', 'current', 'both'), default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=100, metavar='N', help='number of trajectories')
    parser.add_argument('--num_steps', '-t', type=int, default=100, metavar='T', help='maximum number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    parser.add_argument('--reset_states_fname', type=str)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_fname', '-o', type=str)
    args = parser.parse_args()

    # check parsing
    if args.env != 'designed':
        if args.feature_type is not None or args.filter_features is not None:
            raise ValueError("feature_type nor filter_features can only be used "
                             "for the hand-designed feature environment.")
    if args.use_3d_pol:
        if args.interaction_matrix_type is not None:
            raise ValueError("interaction_matrix_type can only be used for "
                             "servoing policy that minimizes the error of "
                             "projected 2d points")

    # actions are translation and angular speed (angular velocity constraint to the (0, 0, 1) axis)
    action_space = TranslationAxisAngleSpace(low=[-20, -10, -10, -np.pi/2],
                                             high=[20, 10, 10, np.pi/2],
                                             axis=[0, 0, 1])
    if args.reset_states_fname is None:
        reset_states = [None] * args.num_trajs
        car_model_names = None
    else:
        with open(args.reset_states_fname, 'rb') as states_file:
            reset_states = pickle.load(states_file)
            if isinstance(reset_states, dict):
                car_model_names = reset_states['car_model_name']
                reset_states = reset_states['reset_states']
            else:
                if os.path.basename(args.reset_states_fname) == 'reset_states.pkl':
                    car_model_names = ['camaro2', 'mazda6', 'sport', 'kia_rio_blue', 'kia_rio_red', 'kia_rio_white']
                elif os.path.basename(args.reset_states_fname) == 'reset_states_hard.pkl':
                    car_model_names = ['kia_rio_silver', 'kia_rio_yellow', 'mitsubishi_lancer_evo']
                else:
                    car_model_names = None
    camera_size, camera_hfov = putil.scale_crop_camera_parameters((640, 480), 60.0, scale_size=0.5, crop_size=(128,) * 2)
    if args.env == 'bbox':
        env = BboxSimpleQuadPanda3dEnv(action_space, car_model_names=car_model_names,
                                       camera_size=camera_size, camera_hfov=camera_hfov)
        env = ServoingEnv(env, max_time_steps=args.num_steps)
    elif args.env == 'bbox3d':
        env = Bbox3dSimpleQuadPanda3dEnv(action_space, car_model_names=car_model_names,
                                         camera_size=camera_size, camera_hfov=camera_hfov)
        env = ServoingEnv(env, max_time_steps=args.num_steps)
    elif args.env == 'designed':
        env = ServoingDesignedFeaturesSimpleQuadPanda3dEnv(action_space,
                                                           feature_type=args.feature_type,
                                                           filter_features=args.filter_features,
                                                           max_time_steps=args.num_steps,
                                                           car_model_names=car_model_names,
                                                           camera_size=camera_size, camera_hfov=camera_hfov)
    else:
        raise ValueError('Invalid environment option %s' % args.env)
    env = NormalizedEnv(env)

    if args.use_3d_pol:
        pol = Point3dBasedServoingPolicy(env, lambda_=args.lambda_, use_car_dynamics=args.use_car_dynamics)
    else:
        pol = PointBasedServoingPolicy(env, lambda_=args.lambda_, interaction_matrix_type=args.interaction_matrix_type, use_car_dynamics=args.use_car_dynamics)

    if args.verbose:
        errors_header_format = '{:>30}{:>15}'
        errors_row_format = '{:>30}{:>15.4f}'
        print(errors_header_format.format('(traj_iter, step_iter)', 'reward'))
    done = False
    discounted_returns = []
    for traj_iter, reset_state in zip(range(args.num_trajs), reset_states):  # whichever is shorter
        np.random.seed(traj_iter)
        if args.verbose:
            print('=' * 45)
        obs = env.reset(reset_state)
        rewards = []
        for step_iter in range(args.num_steps):
            try:
                if args.visualize:
                    target_points_2d = putil.project(env.camera_sensor.lens, obs['target_points'])
                    target_points_xy = putil.points2d_to_xy(env.camera_sensor.lens, target_points_2d)
                    points_2d = putil.project(env.camera_sensor.lens, obs['points'])
                    points_xy = putil.points2d_to_xy(env.camera_sensor.lens, points_2d)

                    vis_image = obs['image'].copy()  # cv2 complains if the image is used directly
                    if args.env == 'bbox':
                        bbox_min = points_xy[0]
                        bbox_max = points_xy[-1]
                        cv2.rectangle(vis_image, tuple(bbox_min), tuple(bbox_max), (0, 255, 0), 1)
                    for point_xy in points_xy:
                        cv2.circle(vis_image, tuple(point_xy), 4, (0, 255, 0), 1)

                    if args.visualize == 1:
                        for target_point_xy in target_points_xy:
                            cv2.circle(vis_image, tuple(target_point_xy), 4, (0, 0, 255), 1)
                    else:
                        offset_xy = np.array([0, vis_image.shape[0]])
                        vis_image = np.vstack([vis_image, obs['target_image']])
                        for point_xy, target_point_xy in zip(points_xy, target_points_xy):
                            cv2.circle(vis_image, tuple(target_point_xy + offset_xy), 4, (0, 255, 0), 1)
                            cv2.line(vis_image, tuple(point_xy), tuple(target_point_xy + offset_xy), (0, 0, 255), 1)

                    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow("image", vis_image)

                    env.render()

                    key = cv2.waitKey(10)
                    key &= 255
                    if key == 27 or key == ord('q'):
                        print("Pressed ESC or q, exiting")
                        done = True

                action = pol.act(obs)

                obs, reward, episode_done, _ = env.step(action)
                rewards.append(reward)
                if args.verbose:
                    print(errors_row_format.format(str((traj_iter, step_iter)), reward))
                if done or episode_done:
                    break
            except KeyboardInterrupt:
                break
        discounted_return = np.array(rewards).dot(args.gamma ** np.arange(len(rewards)))
        if args.verbose:
            print('-' * 45)
            print(errors_row_format.format('discounted return', discounted_return))
        discounted_returns.append(discounted_return)
        if done:
            break
    if args.verbose:
        print('=' * 45)
        print(errors_row_format.format('mean discounted return', np.mean(discounted_returns)))

    if args.output_fname:
        import csv
        with open(args.output_fname, 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([str(args.lambda_),
                             str(np.mean(discounted_returns)),
                             str(np.std(discounted_returns) / np.sqrt(len(discounted_returns)))] +
                            [str(discounted_return) for discounted_return in discounted_returns])


if __name__ == '__main__':
    main()
