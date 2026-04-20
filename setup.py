from setuptools import setup
from Cython.Build import cythonize
import numpy as np

extensions = cythonize(
    "src/wavesst/_core/*.pyx",
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
    },
    annotate=True,
)

setup(ext_modules=extensions, include_dirs=[np.get_include()])
