#!/usr/bin/env python

import runpy
from distutils.core import setup


__version__ = runpy.run_path("medico_sam/__version__.py")["__version__"]


setup(
    name='medico_sam',
    version="0.0.1",
    description='MedicoSAM: Segment Anything for Biomedical Images',
    author=['Anwai Archit', 'Constantin Pape'],
    url='https://user.informatik.uni-goettingen.de/~pape41/',
    packages=['medico_sam'],
)
