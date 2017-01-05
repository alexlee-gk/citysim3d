from panda3d.core import Point2, Point3
from panda3d.core import BoundingBox, BoundingHexahedron
from panda3d.core import VirtualFileSystem


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


# python implementation of the VirtualFileSystem method from here:
# https://github.com/panda3d/panda3d/blob/master/panda/src/express/virtualFileSystem.cxx
def parse_options(options):
    flags = 0
    pw = ''
    for option in options.split(','):
        if option == '0' or not option:
            pass
        elif option == 'ro':
            flags |= VirtualFileSystem.MFReadOnly
        elif option.startswith('pw:'):
            pw = option[3:]
        else:
            raise ValueError('Invalid option on vfs-mount: %s' % option)
    return flags, pw
