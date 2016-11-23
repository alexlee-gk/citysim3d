import numpy as np
from panda3d.core import loadPrcFile
from citysim3d.envs import GeometricCarPanda3dEnv
from citysim3d.spaces import BoxSpace


loadPrcFile('config.prc')


def main():
    # actions are forward acceleration and lateral velocity
    action_space = BoxSpace(low=np.array([0.0, -1.0]),
                            high=np.array([0.0, 1.0]))
    env = GeometricCarPanda3dEnv(action_space)

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
