#!/usr/bin/env python
import os
import subprocess
import argparse
import urllib2


assert "CITYSIM3D_DIR" in os.environ

parser = argparse.ArgumentParser()
parser.add_argument("--rsync", action="store_true")
args = parser.parse_args()


if args.rsync:
    remote_dir = "/var/www/citysim3d/models/"
    local_dir = os.path.expandvars("${CITYSIM3D_DIR}/models/")
    subprocess.check_call("rsync -azvu pabbeel@rll.berkeley.edu:%s %s" % (remote_dir, local_dir), shell=True)
else:
    print "downloading tar file (this might take a while)"
    remote_fname = "http://rll.berkeley.edu/citysim3d/models.tar.gz"
    local_dir = os.path.expandvars("${CITYSIM3D_DIR}/models/")
    local_fname = os.path.join(local_dir, "models.tar.gz")
    urlinfo = urllib2.urlopen(remote_fname)
    with open(local_fname, "w") as fh:
        fh.write(urlinfo.read())

    print "unpacking file"
    # unpack without top-level directory
    subprocess.check_call("tar -xvf %s -C %s --strip-components 1" % (local_fname, local_dir), shell=True)
