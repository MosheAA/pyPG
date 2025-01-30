import numpy
import sys
import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc


###  import LogitWrapper 
###  LogitWrapper
#  -undefined dynamic_lookup
#  -lgsl
#  -fpic   #  build a shared
#  -bundle

#  OMP_THREAD_NUM

#  use --user  to install in
#  to specify compiler, maybe set CC environment variable
#  or python setup.py build --compiler=g++
incldir = [get_python_inc(plat_specific=1), numpy.get_include(), "pyPG/include/RNG", "/usr/local/include"]

libdir = ["/usr/local/lib"]

gplusplus = "g++"
if sys.platform == "darwin":
    gplusplus = "g++-6"   #  OMP doesn't work if g++

#os.environ["CC"]  = gplusplus
os.environ["CXX"] = gplusplus

##  Handle OPENMP switch here
#  http://stackoverflow.com/questions/677577/distutils-how-to-pass-a-user-defined-parameter-to-setup-py
USE_OPENMP = False
#  -fPIC meaningless in osx
#extra_compile_args = ["-fPIC", "-bundle", "-undefined dynamic_lookup", "-shared"]
extra_compile_args = ["-fPIC", "-shared"]
extra_link_args    = ["-lgsl", "-lgslcblas"]

if "--use_openmp" in sys.argv:
    USE_OPENMP = True
    extra_compile_args.extend(["-fopenmp", "-DUSE_OPEN_MP"])
    extra_link_args.append("-lgomp")
    iop = sys.argv.index("--use_openmp")
    sys.argv.pop(iop)

extra_compile_args.extend(["-DNTHROW"])  #  Kensuke Arai needed to do this on MacBook Air using g++-6.  In PolyaGamma.cpp, block with std::invalid_argument causes error like "Symbol not found: __ZNSt16invalid_argumentC1EPKc" on import of pyPG.  


#  may also need to set $LD_LIBRARY_PATH in order to use shared libgsl


sources=['pyPG/PLogitWrapper.cpp', 
         'pyPG/PolyaGamma.cpp', 
         'pyPG/PolyaGammaAlt.cpp', 
         'pyPG/PolyaGammaSP.cpp', 
         'pyPG/LogitWrapper.cpp', 
#         'pyPG/ParallelWrapper.cpp', 
         'pyPG/FSF_nmix.cpp', 
         'pyPG/InvertY.cpp', 
         'pyPG/InvertY2.cpp',
         'pyPG/HHWrapper.cpp',
         'pyPG/include/RNG/GRNG.cpp', 
         'pyPG/include/RNG/RNG.cpp']

#  Output to be named _LogitWrapper.so
module1 = Extension('pyPG/pyPG_cpp_module',
                    libraries = ['gsl'],
                    include_dirs=incldir,
                    library_dirs=libdir,
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,  #  linker args
                    sources=sources)
setup(
    name='pyPG',
    version='0.93',
    packages=['pyPG'],
    ext_modules=[module1],
    author="Kensuke Arai, Jesse Windell, Tim Whalen",
    description="Python Wrapper for Jesse Windell's Polya Gamma sampler",
    email="kensuke.y.arai@gmail.com",
    url="http://github.com/AraiKensuke/pyPG"
)
