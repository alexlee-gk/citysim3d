import numpy as np
from panda3d.core import loadPrcFile
import cv2
from citysim3d.envs import SimpleQuadPanda3dEnv
from citysim3d.spaces import TranslationAxisAngleSpace


loadPrcFile('config.prc')


def main():
    # actions are translation and angular speed (angular velocity constraint to the (0, 0, 1) axis)
    action_space = TranslationAxisAngleSpace(low=[-10, -10, -10, -np.pi/4],
                                             high=[10, 10, 10, np.pi/4],
                                             axis=[0, 0, 1])
    env = SimpleQuadPanda3dEnv(action_space, sensor_names=['image', 'depth_image'])

    num_trajs = 10
    num_steps = 100
    for traj_iter in range(num_trajs):
        env.reset()
        for step_iter in range(num_steps):
            action = action_space.sample()
            env.step(action)
            observations = env.observe()
            image, depth_image = observations

            # convert BGR image to RGB image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("image", image)

            # rescale depth image to be between 0 and 255
            depth_scale = depth_image.max() - depth_image.min()
            depth_offset = depth_image.min()
            depth_image = np.clip((depth_image - depth_offset) / depth_scale, 0.0, 1.0)
            depth_image = (255.0 * depth_image).astype(np.uint8)
            cv2.imshow("depth image", depth_image)

            env.render()

            key = cv2.waitKey(10)
            key &= 255
            if key == 27 or key == ord('q'):
                print("Pressed ESC or q, exiting")


if __name__ == '__main__':
    main()
