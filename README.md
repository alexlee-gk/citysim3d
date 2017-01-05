# CitySim3D
Simulated environments that can be used to test reinforcement learning problems involving vision.

There are two main environments, both in an urban setting:
- Car environment: the action space consists of forward acceleration and lateral velocity (parameterized with respect to the road)
- Quadcopter environment: the action space consists of linear velocity and angular velocity

Currently, neither of these use realistic physics. For better physics simulation, you can implement your own dynamics or use Panda3D's python bindings to Bullet Physics.

You can get RGB and depth images from the agent's camera.

This simulator was used in the paper <a href="https://openreview.net/forum?id=r1YNw6sxg" target="_blank">Learning Visual Servoing with Deep Features and Fitted Q-Iteration</a>.

![Alt Text](http://rll.berkeley.edu/citysim3d/screenshot_top.gif)
![Alt Text](http://rll.berkeley.edu/citysim3d/screenshot_back.gif)

## Install Panda3D from source and link to specific python installation

You can use any python installation you might already have (i.e. you don't need to use pyenv). Regardless of what python installation you use, you should make sure to replace the paths with the right ones in the sample commands below.

### Set up a new python environment using pyenv

Install desired version of python 2 (e.g. 2.7.12). Make sure to use the `--enable-shared` flag to generate python shared libraries, which will later be linked to.
```
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 2.7.12
```

### Install Panda3D in Ubuntu 14.04 and Ubuntu 16.04
```
git clone git@github.com:panda3d/panda3d.git
cd panda3d
pyenv local 2.7.12
python makepanda/makepanda.py --everything --installer --threads 4
sudo dpkg -i panda3d1.10_1.10.0_amd64.deb
```
Create a `panda3d.pth` file so that the Panda3d libraries can be found.
```
echo /usr/share/panda3d >> ~/.pyenv/versions/2.7.12/lib/python2.7/site-packages/panda3d.pth
echo /usr/lib/x86_64-linux-gnu/panda3d >> ~/.pyenv/versions/2.7.12/lib/python2.7/site-packages/panda3d.pth
```

#### Troubleshooting
If the latest source of Panda3d doesn't build successfully, try using one of the stable versions.
```
git checkout tags/v1.9.3
```

### Install Panda3D in MacOS Sierra
```
git clone git@github.com:panda3d/panda3d.git
cd panda3d
pyenv local 2.7.12
```
Replace `elif GetTarget() == 'darwin'` with `elif False` in `makepanda/makepandacore.py` so that Apple's copy of Python is not used, as described in <a href="https://www.panda3d.org/forums/viewtopic.php?f=5&t=18331" target="_blank">here</a>.
```
sed -i -- "s/elif GetTarget() == 'darwin'/elif False/g" makepanda/makepandacore.py
```
Build with an specific include and library directory for python (also e.g. without Maya). Install using `*.dmg` file and follow its installation instructions.
```
python makepanda/makepanda.py --everything --installer --threads 4 \
  --python-incdir ~/.pyenv/versions/2.7.12/include \
  --python-libdir ~/.pyenv/versions/2.7.12/lib \
  --no-maya6 --no-maya65 --no-maya7 --no-maya8 --no-maya85 --no-maya2008 --no-maya2009 --no-maya2010 --no-maya2011 --no-maya2012 --no-maya2013 --no-maya20135 --no-maya2014 --no-maya2015 --no-maya2016 --no-maya2016
open Panda3D-1.10.0.dmg
```
Create a `panda3d.pth` file so that the Panda3d libraries can be found.
```
echo /Developer/Panda3D/ >> ~/.pyenv/versions/2.7.12/lib/python2.7/site-packages/panda3d.pth
echo /Developer/Panda3D/bin >> ~/.pyenv/versions/2.7.12/lib/python2.7/site-packages/panda3d.pth
```

## Install CitySim3D
```
git clone git@github.com:alexlee-gk/citysim3d.git
cd citysim3d
pip install -r requirements.txt
```
Define the environment variable `CITYSIM3D_DIR` to be this directory and add it to the `PYTHONPATH`
```
export CITYSIM3D_DIR=path/to/citysim3d
export PYTHONPATH=$CITYSIM3D_DIR:$PYTHONPATH
```

## Download the 3D models
Run the `models/download.py` script to download all the model files. 

The city and skybox models are under <a href="https://3drt.com/license.htm" target="_blank">this license</a> and those files are encrypted inside `models.mf`. The original city and skybox models came from <a href="http://3drt.com/store/environments/megacity-construction-kit.html" target="_blank">here</a>.
