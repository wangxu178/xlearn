from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [Extension('_kmeans_c', ['_kmeans_c.pyx', 'kmeans_c.c'])]
setup(name='myAI', ext_modules=cythonize(extensions))
