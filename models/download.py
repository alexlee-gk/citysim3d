#!/usr/bin/env python
import os
import subprocess
import argparse
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen
import tarfile


assert "CITYSIM3D_DIR" in os.environ

parser = argparse.ArgumentParser()
parser.add_argument("--rsync", action="store_true")
args = parser.parse_args()

local_dir = os.path.expandvars("${CITYSIM3D_DIR}")
if args.rsync:
    remote_files = "/var/www/citysim3d/models{,.mf}"
    subprocess.check_call("rsync -azvu pabbeel@rll.berkeley.edu:%s %s" % (remote_files, local_dir), shell=True)
else:
    print("downloading tar file (this might take a while)")
    remote_fname = "http://rll.berkeley.edu/citysim3d/models.tar.gz"
    local_fname = os.path.join(local_dir, "models.tar.gz")
    urlinfo = urlopen(remote_fname)
    with open(local_fname, "wb") as fh:
        fh.write(urlinfo.read())

    print("unpacking file")
    with tarfile.open(local_fname) as tar:
        tar.extractall(local_dir)
