import os
import numpy as np
from panda3d.core import loadPrcFile
from panda3d.core import TextNode
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from citysim3d.envs import SimpleQuadPanda3dEnv
from citysim3d.spaces import BoxSpace
import citysim3d.utils.transformations as tf


assert "CITYSIM3D_DIR" in os.environ
loadPrcFile(os.path.expandvars('${CITYSIM3D_DIR}/config.prc'))


def add_instructions(app, pos, msg):
    return OnscreenText(text=msg, style=1, fg=(1, 1, 1, 1), scale=.05,
                        shadow=(0, 0, 0, 1), parent=app.a2dTopLeft,
                        pos=(0.08, -pos - 0.04), align=TextNode.ALeft)


def as_one(boolean):
    """ Maps True to 1 and False to -1 """
    boolean = not not boolean  # ensure it's a boolean
    return 2 * int(boolean) - 1


class CustomSimpleQuadPanda3dEnv(SimpleQuadPanda3dEnv):
    """
    Same as SimpleQuadPanda3dEnv except that actions are directly mapped to
    the car's actions and the quad is constrained to move behind the car.
    """
    def __init__(self, *args, **kwargs):
        super(CustomSimpleQuadPanda3dEnv, self).__init__(*args, **kwargs)

        # change the car's speed limit
        self.car_env.speed_offset_space.low[0] = 0.0
        self.car_env.speed_offset_space.high[0] = np.inf

    @property
    def action_space(self):
        return self.car_env.action_space

    def step(self, action):
        # update the angle of the propellers (for rendering purposes)
        if self.prop_rpm:
            self.prop_angle += (self.prop_rpm * 2 * np.pi / 60) * self.dt
            self.prop_angle -= 2 * np.pi * np.floor(self.prop_angle / (2 * np.pi))
            for quad_prop_id, quad_prop_local_node in enumerate(self.quad_prop_local_nodes):
                is_ccw = quad_prop_id in (1, 2)
                angle = self.prop_angle if is_ccw else -self.prop_angle
                quad_prop_local_node.setQuat(tuple(tf.quaternion_about_axis(angle, np.array([0, 0, 1]))))

        self.car_env.step(action)

        # set the position of the quad to be behind the car
        car_T = tf.pose_matrix(self.car_node.getQuat(), self.car_node.getPos())
        quad_pos = car_T[:3, 3] + car_T[:3, :3].dot(np.array([0., -4., 3.]) * 4)
        # set the rotation of the quad to be the rotation of the car projected so that the z-axis is up
        axis = np.cross(car_T[:3, 2], np.array([0, 0, 1]))
        angle = tf.angle_between_vectors(car_T[:3, 2], np.array([0, 0, 1]))
        if np.isclose(angle, 0.0):
            project_T = np.eye(4)
        else:
            project_T = tf.rotation_matrix(angle, axis)
        quad_T = project_T.dot(car_T)
        quad_quat = tf.quaternion_from_matrix(quad_T[:3, :3])
        tightness = 0.1
        self.quad_node.setPosQuat(tuple((1 - tightness) * np.array(self.quad_node.getPos()) + tightness * quad_pos), tuple(quad_quat))

        return self.observe(), None, False, dict()

    def reset(self):
        """
        Same as SimpleQuadPanda3dEnv.reset except that the car's speed is
        always set to 10.0"""
        # reset to given or random state...
        super(CustomSimpleQuadPanda3dEnv, self).reset()
        state = self.car_env.get_state()
        # ... but then set the car's speed to 10.0
        state[0] = 10.0
        self.car_env.set_state(state)
        return self.observe()


def main():
    # actions are forward acceleration and lateral velocity
    action_space = BoxSpace(low=np.array([-1.0, -1.0]),
                            high=np.array([1.0, 1.0]))
    car_model_names = ['camaro2', 'kia_rio_blue', 'kia_rio_red',
                       'kia_rio_silver', 'kia_rio_white', 'kia_rio_yellow',
                       'mazda6', 'mitsubishi_lancer_evo', 'sport']
    env = CustomSimpleQuadPanda3dEnv(action_space,
                                     sensor_names=[],  # empty sensor_names means no observations
                                     car_model_names=car_model_names)

    num_camera_modes = 3
    key_map = dict(left=False, right=False, up=False, down=False, camera_pressed=False, camera_mode=0)

    def step(task):
        forward_acceleration = as_one(key_map['up']) - as_one(key_map['down'])
        lateral_velocity = as_one(key_map['right']) - as_one(key_map['left'])
        action = np.array([forward_acceleration, lateral_velocity])
        env.step(action)

        if key_map['camera_pressed']:
            key_map['camera_mode'] = (key_map['camera_mode'] + 1) % num_camera_modes
        if key_map['camera_mode'] == 0:
            env.app.cam.reparentTo(env.app.render)
            env.app.cam.setQuat(tuple(tf.quaternion_about_axis(-np.pi / 2, np.array([1, 0, 0]))))
            env.app.cam.setPos(tuple(np.array(env.car_node.getPos()) + np.array([0., 0., 100.])))
        elif key_map['camera_mode'] in (1, 2):
            if key_map['camera_pressed']:
                tightness = 1.0
            else:
                tightness = 0.1
            if key_map['camera_mode'] == 1:
                target_node = env.car_node
                offset = np.array([0., -4., 3.]) * 3
            else:
                target_node = env.quad_node
                offset = np.array([0., -4., 3.]) * .5
            target_T = tf.pose_matrix(target_node.getQuat(), target_node.getPos())
            target_camera_pos = target_T[:3, 3] + target_T[:3, :3].dot(offset)
            env.app.cam.setPos(tuple((1 - tightness) * np.array(env.app.cam.getPos()) + tightness * target_camera_pos))
            env.app.cam.lookAt(target_node)
        else:
            env.app.cam.reparentTo(env.car_node)
            env.app.cam.setQuat((1, 0, 0, 0))
            env.app.cam.setPos(tuple(np.array([0, 1, 2])))  # slightly in front of the car
        key_map['camera_pressed'] = False
        return Task.cont

    env.app.taskMgr.add(step, "step")

    add_instructions(env.app, 0.06, "[ESC]: Quit")
    add_instructions(env.app, 0.12, "[R]: Reset environment")
    add_instructions(env.app, 0.18, "[Left Arrow]: Move car left")
    add_instructions(env.app, 0.24, "[Right Arrow]: Move car right")
    add_instructions(env.app, 0.30, "[Up Arrow]: Accelerate the car")
    add_instructions(env.app, 0.36, "[Down Arrow]: Decelerate the car")
    add_instructions(env.app, 0.42, "[C]: Toggle camera mode")
    add_instructions(env.app, 0.48, "[S]: Take screenshot")

    env.app.accept('r', env.reset)
    env.app.accept('arrow_left', key_map.update, [[('left', True)]])
    env.app.accept('arrow_left-up', key_map.update, [[('left', False)]])
    env.app.accept('arrow_right', key_map.update, [[('right', True)]])
    env.app.accept('arrow_right-up', key_map.update, [[('right', False)]])
    env.app.accept('arrow_up', key_map.update, [[('up', True)]])
    env.app.accept('arrow_up-up', key_map.update, [[('up', False)]])
    env.app.accept('arrow_down', key_map.update, [[('down', True)]])
    env.app.accept('arrow_down-up', key_map.update, [[('down', False)]])
    env.app.accept('c-up', key_map.update, [[('camera_pressed', True)]])
    env.app.accept('s-up', env.app.screenshot)

    env.reset()
    env.app.run()


if __name__ == '__main__':
    main()
