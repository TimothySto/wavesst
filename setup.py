from setuptools import setup
from glob import glob

pyx_files = glob("src/wavesst/_core/*.pyx")

if pyx_files:
    from Cython.Build import cythonize
    import numpy as np
    extensions = cythonize(
        pyx_files,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
        annotate=True,
    )
    setup(ext_modules=extensions, include_dirs=[np.get_include()])
else:
    setup()
