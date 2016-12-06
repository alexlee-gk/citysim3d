import sys
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties, FrameBufferProperties
from panda3d.core import GraphicsPipe, GraphicsEngine, GraphicsOutput
from panda3d.core import Texture
from panda3d.core import BitMask32
from citysim3d.envs import Env
import citysim3d.utils.transformations as tf


class Panda3dEnv(Env):
    def __init__(self, app=None, dt=None):
        self.app = app or ShowBase()
        self.app.accept('escape', sys.exit)
        self.root_node = self.app.render

        self._dt = 0.1 if dt is None else dt

        # setup visualization camera
        vfov = 45
        hfov = vfov * float(self.app.win.size[0]) / float(self.app.win.size[1])
        self.app.camLens.setFov(hfov, vfov)
        self.app.camLens.set_near_far(0.01, 10000.0)  # 1cm to 10km

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
    def __init__(self, base, color=True, depth=False, size=(640, 480), near_far=(0.01, 10000.0), hfov=60):
        self.size = size
        winprops = WindowProperties.size(*self.size)
        winprops.setTitle('Camera Sensor')
        fbprops = FrameBufferProperties()
        # Request 8 RGB bits, 8 alpha bits, and a depth buffer.
        fbprops.setRgbColor(True)
        fbprops.setRgbaBits(8, 8, 8, 8)
        fbprops.setDepthBits(24)
        self.graphics_engine = GraphicsEngine(base.pipe)

        window_type = base.config.GetString('window-type', 'onscreen')
        flags = GraphicsPipe.BFFbPropsOptional
        if window_type == 'onscreen':
            flags = flags | GraphicsPipe.BFRequireWindow
        elif window_type == 'offscreen':
            flags = flags | GraphicsPipe.BFRefuseWindow

        self.buffer = self.graphics_engine.makeOutput(
            base.pipe, "camera sensor buffer", -100,
            fbprops, winprops, flags)

        if not color and not depth:
            raise ValueError("At least one of color or depth should be True")
        if color:
            self.color_tex = Texture("color_texture")
            self.buffer.addRenderTexture(self.color_tex, GraphicsOutput.RTMCopyRam,
                                         GraphicsOutput.RTPColor)
        else:
            self.color_tex = None
        if depth:
            self.depth_tex = Texture("depth_texture")
            self.buffer.addRenderTexture(self.depth_tex, GraphicsOutput.RTMCopyRam,
                                         GraphicsOutput.RTPDepth)
        else:
            self.depth_tex = None

        self.cam = base.makeCamera(self.buffer, scene=base.render, camName='camera_sensor')
        self.lens = self.cam.node().getLens()
        self.lens.setFov(hfov)
        self.lens.setFilmSize(*size)  # this also defines the units of the focal length
        self.lens.setNearFar(*near_far)

    @property
    def focal_length(self):
        # same as
        # size[1] / (2. * np.tan(np.deg2rad(lens.getVfov()) / 2.))
        return self.lens.getFocalLength()

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
            image = image[..., :-1]  # remove alpha channel; if alpha values are needed, set alpha bits to 8
            images.append(image)

        if self.depth_tex:
            depth_data = self.depth_tex.getRamImage()
            if sys.version_info < (3, 0):
                depth_data = depth_data.get_data()
            depth_image_size = self.depth_tex.getYSize() * self.depth_tex.getXSize() * self.depth_tex.getNumComponents()
            if len(depth_data) == 2 * depth_image_size:
                dtype = np.float16
            elif len(depth_data) == 3 * depth_image_size:
                dtype = np.float24
            elif len(depth_data) == 4 * depth_image_size:
                dtype = np.float32
            else:
                raise ValueError("Depth data has %d bytes but the size of the depth image is %d" % (len(depth_data), depth_image_size))
            depth_image = np.frombuffer(depth_data, dtype)
            depth_image.shape = (self.depth_tex.getYSize(), self.depth_tex.getXSize(), self.depth_tex.getNumComponents())
            depth_image = np.flipud(depth_image)
            depth_image = depth_image.astype(np.float32, copy=False)  # copy only if necessary
            images.append(depth_image)

        return tuple(images)


class Panda3dMaskCameraSensor(object):
    def __init__(self, base, hidden_nodes, size=(640, 480), near_far=(0.01, 10000.0), hfov=60):
        # renders everything
        self.all_camera_sensor = Panda3dCameraSensor(base, color=False, depth=True,
                                                     size=size, near_far=near_far, hfov=hfov)
        # renders the non-hidden nodes
        self.visible_camera_sensor = Panda3dCameraSensor(base, color=False, depth=True,
                                                         size=size, near_far=near_far, hfov=hfov)
        self.cam = (self.all_camera_sensor.cam, self.visible_camera_sensor.cam)

        non_hidden_mask = BitMask32(0x3FFFFFFF)
        hidden_mask = BitMask32(0x40000000)

        self.all_camera_sensor.cam.node().setCameraMask(non_hidden_mask)
        self.visible_camera_sensor.cam.node().setCameraMask(hidden_mask)

        for hidden_node in hidden_nodes:
            hidden_node.show(non_hidden_mask | hidden_mask)
            hidden_node.hide(hidden_mask)

    def observe(self):
        depth_image, = self.all_camera_sensor.observe()
        non_hidden_depth_image, = self.visible_camera_sensor.observe()
        mask = (np.logical_and(non_hidden_depth_image < 1,
                               non_hidden_depth_image <= depth_image
                               ) * 255).astype(np.uint8)
        return mask, depth_image, non_hidden_depth_image
