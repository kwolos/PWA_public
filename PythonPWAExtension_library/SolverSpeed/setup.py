#%%
from distutils.core import setup, Extension
import numpy

ext = Extension('PythonPWAExtension', 
				sources=['./C++ source code/PythonPWAspeed.cpp'],
				libraries=['m', 'gsl', 'gslcblas'],	# add these libraries 
				extra_compile_args=['-Wall', '-c', '-I/usr/include', '-fopenmp'], # add args which are essential for compiling
				extra_link_args=['-L/usr/lib/', '-fopenmp'],
				include_dirs=[numpy.get_include()],
				depends=['./C++ source code/biffurcationPoint.h', 
						 './C++ source code/inflowFunctions.h',
						 './C++ source code/params.h',
						 './C++ source code/terminalEnd.h',
						 './C++ source code/updateInterior.h',
						 './C++ source code/vessel.h']) 


setup(name = 'PythonPWAExtension', 
	  version = '0.9.6',  \
   	  ext_modules = [ext])
