#!/usr/bin/env python

import runpy
from setuptools import find_packages, setup


__version__ = runpy.run_path("medico_sam/__version__.py")["__version__"]


setup(
    name='medico_sam',
    version=__version__,
    description='MedicoSAM: Segment Anything for Biomedical Images',
    author=['Anwai Archit', 'Constantin Pape'],
    url='https://user.informatik.uni-goettingen.de/~pape41/',
    packages=find_packages(include=['medico_sam', 'medico_sam.*']),
    license="MIT",
    install_requires=["micro_sam>=1.8.1"],
)
