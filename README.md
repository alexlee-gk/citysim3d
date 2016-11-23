# CitySim3D

## Install Panda3D from source and link to specific python installation

### Set up a new python environment using pyenv

Install desired version of python 2 (e.g. 2.7.12). Make sure to use the `--enable-shared` flag to generate python shared libraries, which will later be linked to.
```
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 2.7.12
```

### Install Panda3D in Ubuntu 14.04
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

### Install Panda3D in MacOS
```
git clone git@github.com:panda3d/panda3d.git
cd panda3d
pyenv local 2.7.12
```
Replace `elif GetTarget() == 'darwin'` with `elif False` in `makepanda/makepandacore.py` so that Apple's copy of Python is not used, as described in [here](https://www.panda3d.org/forums/viewtopic.php?f=5&t=18331).
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

## Download 3D Model
Run the `download.py` script to rsync the files from a (password-protected) remote account.
```
python models/download.py
```
