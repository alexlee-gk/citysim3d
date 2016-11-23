#!/usr/bin/env python
import os
import subprocess

assert "CITYSIM3D_DIR" in os.environ

remote_dir = "/var/www/citysim3d/models_public/"
local_dir = os.path.expandvars("${CITYSIM3D_DIR}/models/")

subprocess.check_call("rsync -azvu pabbeel@rll.berkeley.edu:%s %s" % (remote_dir, local_dir), shell=True)
