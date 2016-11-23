###########################################################
###                                                     ###
### Panda3D Configuration File -  User-Editable Portion ###
###                                                     ###
###########################################################

# These control the placement and size of the default rendering window.
# A value of -2 for the origin means to center it on the screen,
# while -1 lets the window manager choose the position.

win-origin -2 -2
win-size 640 480

# These control the amount of output Panda gives for some various
# categories.  The severity levels, in order, are "spam", "debug",
# "info", "warning", and "error"; the default is "info".  Uncomment
# one (or define a new one for the particular category you wish to
# change) to control this output.

notify-level warning
default-directnotify-level warning

# These specify where model files may be loaded from.  You probably
# want to set this to a sensible path for yourself.  $THIS_PRC_DIR is
# a special variable that indicates the same directory as this
# particular config.prc file.

model-path    $CITYSIM3D_DIR/models/megacity-urban-construction-kit
model-path    $CITYSIM3D_DIR/models/vehicles

# Enable/disable performance profiling tool and frame-rate meter

want-pstats            #f
show-frame-rate-meter  #f

# Enable audio using the OpenAL audio library by default:

audio-library-name null

# This option specifies the default profiles for Cg shaders.
# Setting it to #t makes them arbvp1 and arbfp1, since these
# seem to be most reliable. Setting it to #f makes Panda use
# the latest profile available.

basic-shaders-only #f

# window-type offscreen
sync-video #f
texture-anisotropic-degree 8
depth-bits 24
