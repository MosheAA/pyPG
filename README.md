
# pyPG
Sampling Polya-Gamma random variables.  Python wrapper for R/C++ code by Jesse Windle.

### Installation
pyPG is a set of python/cython wrappers around the C++ code originally written by Jesse Windle.  Installation requires the GNU Scientific library `gsl`, found [here](https://www.gnu.org/software/gsl/).  

To install, unpack pyPG in its own directory, and run the install script.

`python setup.py install --use_openmp`

If the C++ compiler doesn't support OpenMP, upgrade to one that does, or simply leave the `--use_openmp` flag off.  

If compile fails, the `incldir` and `libdir` may need to be modified for your environment.




