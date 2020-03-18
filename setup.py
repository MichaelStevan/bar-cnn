# Built-In Imports
import setuptools
from setuptools.extension import Extension

# PyPi Imports
import numpy as np
extensions = [
    Extension(
        'bar_cnn.utils.compute_overlap',
        ['bar_cnn/utils/compute_overlap.pyx'],
        include_dirs=[np.get_include()]
    ),
]

setuptools.setup(
    name='bar-cnn',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/darien-schettler/bar-cnn',
    license='OSI Approved :: Apache Software License',
    author='darienschettler',
    author_email='darien_schettler@hotmail.com',
    maintainer='Darien Schettler',
    maintainer_email='darien.schettler@quantiphi.com',
    # cmdclass={'build_ext': BuildExtension},
    description='Implementation of the 2018 Paper - Detecting Visual Relationships Using Box Attention',
    install_requires=['six', 'tensorflow', 'numpy'],
    entry_points={
        'console_scripts': [
            # 'bar-cnn-train=bar_cnn.bin.train:main',
            # 'bar-cnn-evaluate=bar_cnn.bin.evaluate:main',
        ],
    },
    ext_modules=extensions,
    setup_requires=["cython>=0.28", "numpy>=1.14.0"]
)
