import os
import time
import numpy as np
from panda3d.core import loadPrcFile
from citysim3d.envs import GeometricCarPanda3dEnv
from citysim3d.spaces import BoxSpace

assert "CITYSIM3D_DIR" in os.environ
loadPrcFile(os.path.expandvars('${CITYSIM3D_DIR}/config.prc'))


def main():
    # actions are forward acceleration and lateral velocity
    action_space = BoxSpace(low=np.array([0.0, -1.0]),
                            high=np.array([0.0, 1.0]))
    env = GeometricCarPanda3dEnv(action_space)

    start_time = time.time()
    num_trajs = 10
    num_steps = 100
    for traj_iter in range(num_trajs):
        env.reset()
        for step_iter in range(num_steps):
            action = action_space.sample()
            env.step(action)
            env.render()
    print("average FPS: {}".format(num_trajs * (num_steps + 1) / (time.time() - start_time)))


if __name__ == '__main__':
    main()
