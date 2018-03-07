
# pyPG
Sampling Polya-Gamma random variables.  Python wrapper for R/C++ code by Jesse Windle.

### Installation
pyPG is a set of python/C++ wrappers around the C++ code originally written by Jesse Windle.  Instead of being written in using cython, this is written in C++ with an include of `<Python.h>` to allow python objects to interface with the C++.  No particular reason for this decision, but this is just the first way I learned about to interface C++ code with python, before learning about cython.  

Installation requires the GNU Scientific library `gsl`, found [here](https://www.gnu.org/software/gsl/).  Be sure to note where you choose to install the include files and libraries, because those values will be needed in the setup.py script.

To install, unpack pyPG in its own directory, say `${HOME}/pyPG`, and run the install script.

```
cd ${HOME}/pyPG
python setup.py install --use_openmp
```
This installs pyPG in python library under site-packages, which on my system is 
`/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages`

If you don't want to install in the system-wide python installation, do 

`python setup.py build --use_openmp`

This installs pyPG in python library under the `${HOME}/pyPG/build/..../` directory.  There will be a directory called pyPG, with contents `__init__.py`, and `_pyPG.so`.  You can then point the environment variable `PYTHONPATH` to this directory, or move the `pyPG` directory to any desired location, and included that location in the `PYTHONPATH` environment variable.

If the C++ compiler doesn't support OpenMP, upgrade to one that does, or simply leave the `--use_openmp` flag off.  

If compile fails, the `incldir` and `libdir` may need to be modified for your environment.

