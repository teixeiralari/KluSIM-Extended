from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
                "src.vptree.vptree",
                ["src/vptree/vptree.pyx"],
                include_dirs=[np.get_include()],
            ),
    Extension(
                "src.metrics.distance_metrics",
                ["src/metrics/distance_metrics.pyx"],
                include_dirs=[np.get_include()],
            ),
    Extension(
                "src.utils.helpers",
                ["src/utils/helpers.pyx"],
                include_dirs=[np.get_include()],
            ),
    Extension(
                "klusim",
                ["src/klusim.pyx"],
                include_dirs=[np.get_include()],
            ),
    Extension(
                "sfkm",
                ["src/sfkm.pyx"],
                include_dirs=[np.get_include()],
            ),
    Extension(
                "fames",
                ["src/fames.pyx"],
                include_dirs=[np.get_include()],
            ),
    Extension(
                "kmedoids_initialization",
                ["src/utils/kmedoids_initialization.pyx"],
                include_dirs=[np.get_include()],
            ),
    Extension(
                "fasterpam",
                ["src/fasterpam.pyx"],
                include_dirs=[np.get_include()],
            )         
]

setup(
    ext_modules=cythonize(extensions)
)