import numpy as np
import cv2
from panda3d.core import Point2, Point3
from panda3d.core import BoundingBox, BoundingHexahedron


def make_bounds(lens, scale_size=None, crop_size=None):
    """
    Allocates and returns a new BoundingVolume that encloses the frustum used
    for this kind of lens, if possible.  If a suitable bounding volume cannot
    be created, returns None.

    Same as Lens's make_bounds method except that the frustrum could be smaller
    by specifying a scale_size or crop_size.

    Original implementation of make_bounds is in here:
    https://github.com/panda3d/panda3d/blob/master/panda/src/gobj/lens.cxx
    """
    fll = Point3()
    flr = Point3()
    ful = Point3()
    fur = Point3()
    nll = Point3()
    nlr = Point3()
    nul = Point3()
    nur = Point3()

    film_size = lens.getFilmSize()
    scale_size = scale_size or 1.0
    crop_size = tuple(crop_size) or film_size
    ll = Point2(-crop_size[0] / scale_size / film_size[0], -crop_size[1] / scale_size / film_size[1])
    lr = Point2(+crop_size[0] / scale_size / film_size[0], -crop_size[1] / scale_size / film_size[1])
    ul = Point2(-crop_size[0] / scale_size / film_size[0], +crop_size[1] / scale_size / film_size[1])
    ur = Point2(+crop_size[0] / scale_size / film_size[0], +crop_size[1] / scale_size / film_size[1])

    # Upper left.
    if not lens.extrude(ul, nul, ful):
        return None

    # Upper right.
    if not lens.extrude(ur, nur, fur):
        return None

    # Lower right.
    if not lens.extrude(lr, nlr, flr):
        return None

    # Lower left.
    if not lens.extrude(ll, nll, fll):
        return None

    return BoundingHexahedron(fll, flr, fur, ful, nll, nlr, nur, nul)


def is_in_view(cam_node, obj_node, scale_size=None, crop_size=None):
    """
    Returns the intersection flag between the camera's frustrum and the
    object's tight bounding box.

    https://www.panda3d.org/forums/viewtopic.php?t=11704

    Intersection flags are defined in here:
    https://github.com/panda3d/panda3d/blob/master/panda/src/mathutil/boundingVolume.h
    """
    lens_bounds = make_bounds(cam_node.node().getLens(), scale_size=scale_size, crop_size=crop_size)
    bounds = BoundingBox(*obj_node.getTightBounds())
    bounds.xform(obj_node.getParent().getMat(cam_node))
    return lens_bounds.contains(bounds)


def project(lens, points3d):
    assert lens.isLinear()
    proj_mat = np.array(lens.getProjectionMat()).T
    points3d = np.asarray(points3d)
    points3d_full = np.c_[points3d, np.ones(len(points3d))]
    points2d_full = points3d_full.dot(proj_mat.T)
    points2d = points2d_full[:, :2] / points2d_full[:, 2][:, None]
    return points2d


def extrude_depth(lens, points2d):
    """
    Uses the depth component of the 3-d result from project() to compute the
    original point in 3-d space corresponding to a particular point on the
    lens.  This exactly reverses project(), assuming the point does fall
    legitimately within the lens.

    If points2d is an iterable of 3-d points, the first two dimensions of each
    point should be in the range (-1,1) in both dimensions, where (0,0) is the
    center of the lens and (-1,-1) is the lower-left corner. The last dimension
    of each point is the depth z from the depth buffer.

    If points2d is a depth map of shape (height, width, 3), the 2d coordinates
    are implicitly defined by the position within (height, width).
    """
    assert lens.isLinear()
    proj_mat_inv = np.array(lens.getProjectionMatInv()).T
    points2d = np.asarray(points2d)
    if points2d.ndim == 2 and points2d.shape[1] == 3:
        points2d_full = np.c_[points2d, np.ones(len(points2d))]
        points3d_full = points2d_full.dot(proj_mat_inv.T)
        points3d = points3d_full[:, :3] / points3d_full[:, 3][:, None]
    elif points2d.ndim == 3 and points2d.shape == (lens.film_size[1], lens.film_size[0], 1):
        x, y = np.meshgrid(np.arange(lens.film_size[0]), np.arange(lens.film_size[1]))
        points2d = np.concatenate([x[..., None], y[..., None], points2d], axis=-1)
        points3d = extrude_depth(lens, points2d.reshape((-1, 3))).reshape(points2d.shape)
    return points3d


def xy_to_points2d(lens, xy):
    c_xy = np.array([lens.film_size[0], lens.film_size[1]]) / 2.0 - 0.5
    points2d = 2.0 * (xy - c_xy) / (np.array(lens.film_size) - 1.0)
    points2d = points2d * np.array([1.0, -1.0])
    return points2d


def points2d_to_xy(lens, points2d):
    c_xy = np.array([lens.film_size[0], lens.film_size[1]]) / 2.0 - 0.5
    points2d = points2d * np.array([1.0, -1.0])
    xy = np.round(points2d * (np.array(lens.film_size) - 1.0) / 2.0 + c_xy).astype(int)
    return xy


def xy_depth_to_XYZ(lens, points_xy, depth_image):
    # normalize between -1.0 and 1.0
    points_2d = xy_to_points2d(lens, points_xy)
    # append z depth to it
    points_z = np.array([cv2.getRectSubPix(depth_image, (1, 1), tuple(point_xy))[0][0] for point_xy in points_xy])
    points_2d = np.c_[points_2d, points_z]
    # extrude to 3d points in the camera's local frame
    points_XYZ = extrude_depth(lens, points_2d)
    return points_XYZ


def scale_crop_camera_parameters(orig_size, orig_hfov, scale_size=None, crop_size=None):
    """
    Returns the parameters (size, hfov) of the camera that renders an image
    which is equivalent to an image that is first rendered from the original
    camera and then scaled by scale_size and center-cropped by crop_size.
    """
    scale_size = scale_size if scale_size is not None else 1.0
    crop_size = crop_size if crop_size is not None else orig_size
    size = crop_size
    hfov = np.rad2deg(2 * np.arctan(np.tan(np.deg2rad(orig_hfov) / 2.) * crop_size[0] / orig_size[0] / scale_size))
    return size, hfov
