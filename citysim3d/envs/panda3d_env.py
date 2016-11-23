import sys
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties, FrameBufferProperties
from panda3d.core import GraphicsPipe, GraphicsEngine, GraphicsOutput
from panda3d.core import Texture
from citysim3d.envs import Env
import citysim3d.utils.transformations as tf


class Panda3dEnv(Env):
    def __init__(self, app=None, dt=None):
        self.app = app or ShowBase()
        self.app.accept('escape', sys.exit)

        self.app.camLens.set_near_far(0.01, 10000.0)  # 1cm to 10km
        self._dt = 0.1 if dt is None else dt

    def render(self):
        self.app.cam.setQuat(tuple(tf.quaternion_about_axis(-np.pi / 2, np.array([1, 0, 0]))))
        self.app.cam.setPos((0, 0, 2000))

        for _ in range(self.app.graphicsEngine.getNumWindows()):
            self.app.graphicsEngine.renderFrame()
        self.app.graphicsEngine.syncFrame()

    @property
    def dt(self):
        return self._dt


class Panda3dCameraSensor(object):
    def __init__(self, base, color=True, depth=False, size=(640, 480), near_far=(0.01, 10000.0), vfov=45):
        self.size = size
        winprops = WindowProperties.size(*self.size)
        winprops.setTitle('Camera Sensor')
        fbprops = FrameBufferProperties()
        fbprops.setRgbColor(1)
        fbprops.setAlphaBits(1)
        fbprops.setDepthBits(1)
        self.graphics_engine = GraphicsEngine(base.pipe)

        window_type = base.config.GetString('window-type', 'onscreen')
        flags = GraphicsPipe.BFFbPropsOptional
        if window_type == 'onscreen':
            flags = flags | GraphicsPipe.BFRequireWindow
        elif window_type == 'offscreen':
            flags = flags | GraphicsPipe.BFRefuseWindow

        self.buffer = self.graphics_engine.makeOutput(
            base.pipe, "camera sensor buffer", 0,
            fbprops, winprops, flags)

        if not color and not depth:
            raise ValueError("At least one of color or depth should be True")
        if color:
            self.color_tex = Texture()
            self.buffer.addRenderTexture(self.color_tex, GraphicsOutput.RTMCopyRam,
                                         GraphicsOutput.RTPColor)
        else:
            self.color_tex = None
        if depth:
            self.depth_tex = Texture()
            self.buffer.addRenderTexture(self.depth_tex, GraphicsOutput.RTMCopyRam,
                                         GraphicsOutput.RTPDepth)
        else:
            self.depth_tex = None

        self.cam = base.makeCamera(self.buffer, scene=base.render, camName='camera_sensor')
        hfov = vfov * float(size[0]) / float(size[1])
        self.cam.node().getLens().setFov(hfov, vfov)
        self.cam.node().getLens().setNearFar(*near_far)

    @property
    def focal_length(self):
        focal_length = self.size[1] / self.cam.node().getLens().getFocalLength()
        # same as
        # self.size[0] / (2. * np.tan(np.deg2rad(self.cam.node().getLens().getHfov()) / 2.))
        return focal_length

    def observe(self):
        for _ in range(self.graphics_engine.getNumWindows()):
            self.graphics_engine.renderFrame()
        self.graphics_engine.syncFrame()

        images = []

        if self.color_tex:
            data = self.color_tex.getRamImageAs('RGBA')
            if sys.version_info < (3, 0):
                data = data.get_data()
            image = np.frombuffer(data, np.uint8)
            image.shape = (self.color_tex.getYSize(), self.color_tex.getXSize(), self.color_tex.getNumComponents())
            image = np.flipud(image)
            image = image[..., :-1]  # remove alpha channel
            images.append(image)

        if self.depth_tex:
            depth_data = self.depth_tex.getRamImage()
            if sys.version_info < (3, 0):
                depth_data = depth_data.get_data()
            depth_image = np.frombuffer(depth_data, np.float32)
            depth_image.shape = (
            self.depth_tex.getYSize(), self.depth_tex.getXSize(), self.depth_tex.getNumComponents())
            depth_image = np.flipud(depth_image)
            images.append(depth_image)

        return tuple(images)