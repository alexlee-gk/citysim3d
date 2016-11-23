import numpy as np
from panda3d.core import loadPrcFile
from citysim3d.envs import SimpleQuadPanda3dEnv
from citysim3d.spaces import TranslationAxisAngleSpace


loadPrcFile('config.prc')


def main():
    # actions are translation and angular speed (angular velocity constraint to the (0, 0, 1) axis)
    action_space = TranslationAxisAngleSpace(low=[-10, -10, -10, -np.pi/4],
                                             high=[10, 10, 10, np.pi/4],
                                             axis=[0, 0, 1])
    env = SimpleQuadPanda3dEnv(action_space)

    num_trajs = 10
    num_steps = 100
    for traj_iter in range(num_trajs):
        env.reset()
        for step_iter in range(num_steps):
            action = action_space.sample()
            env.step(action)
            env.render()


if __name__ == '__main__':
    main()
