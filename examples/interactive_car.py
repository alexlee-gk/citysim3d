import os
import numpy as np
from panda3d.core import loadPrcFile
from panda3d.core import TextNode
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from citysim3d.envs import GeometricCarPanda3dEnv
from citysim3d.spaces import BoxSpace
import citysim3d.utils.transformations as tf


assert "CITYSIM3D_DIR" in os.environ
loadPrcFile(os.path.expandvars('${CITYSIM3D_DIR}/config.prc'))


def add_instructions(app, pos, msg):
    return OnscreenText(text=msg, style=1, fg=(1, 1, 1, 1), scale=.07,
                        shadow=(0, 0, 0, 1), parent=app.a2dTopLeft,
                        pos=(0.08, -pos - 0.04), align=TextNode.ALeft)


def as_one(boolean):
    """ Maps True to 1 and False to -1 """
    boolean = not not boolean  # ensure it's a boolean
    return 2 * int(boolean) - 1


def main():
    # actions are forward acceleration and lateral velocity
    action_space = BoxSpace(low=np.array([-1.0, -1.0]),
                            high=np.array([1.0, 1.0]))
    env = GeometricCarPanda3dEnv(action_space, sensor_names=[])  # empty sensor_names means no observations
    # change the car's speed limit
    env.speed_offset_space.low[0] = 0.0
    env.speed_offset_space.high[0] = np.inf

    def reset():
        # reset to random state...
        env.reset()
        state = env.get_state()
        # ... but then set the speed to 1.0
        state[0] = 1.0
        env.reset(state)

    key_map = dict(left=False, right=False, up=False, down=False, camera_fpv=False)

    def step(task):
        forward_acceleration = as_one(key_map['up']) - as_one(key_map['down'])
        lateral_velocity = as_one(key_map['right']) - as_one(key_map['left'])
        action = np.array([forward_acceleration, lateral_velocity])
        env.step(action)

        if key_map['camera_fpv']:
            env.app.cam.reparentTo(env.car_node)
            env.app.cam.setQuat((1, 0, 0, 0))
            env.app.cam.setPos(tuple(np.array([0, 1, 2])))  # slightly in front of the car
        else:
            env.app.cam.reparentTo(env.app.render)
            env.app.cam.setQuat(tuple(tf.quaternion_about_axis(-np.pi / 2, np.array([1, 0, 0]))))
            env.app.cam.setPos(tuple(np.array(env.car_node.getPos()) + np.array([0., 0., 100.])))
        return Task.cont

    env.app.taskMgr.add(step, "step")

    add_instructions(env.app, 0.08, "[ESC]: Quit")
    add_instructions(env.app, 0.16, "[R]: Reset environment")
    add_instructions(env.app, 0.24, "[Left Arrow]: Move car left")
    add_instructions(env.app, 0.32, "[Right Arrow]: Move car right")
    add_instructions(env.app, 0.40, "[Up Arrow]: Accelerate the car")
    add_instructions(env.app, 0.48, "[Down Arrow]: Deccelerate the car")
    add_instructions(env.app, 0.56, "[C]: Toggle camera mode")
    add_instructions(env.app, 0.64, "[Mouse]: Move main camera")

    env.app.accept('r', reset)
    env.app.accept('arrow_left', key_map.update, [[('left', True)]])
    env.app.accept('arrow_left-up', key_map.update, [[('left', False)]])
    env.app.accept('arrow_right', key_map.update, [[('right', True)]])
    env.app.accept('arrow_right-up', key_map.update, [[('right', False)]])
    env.app.accept('arrow_up', key_map.update, [[('up', True)]])
    env.app.accept('arrow_up-up', key_map.update, [[('up', False)]])
    env.app.accept('arrow_down', key_map.update, [[('down', True)]])
    env.app.accept('arrow_down-up', key_map.update, [[('down', False)]])
    env.app.accept('c-up', lambda: key_map.update([('camera_fpv', not key_map['camera_fpv'])]))  # toggle camera_fpv key

    reset()
    env.app.run()


if __name__ == '__main__':
    main()
