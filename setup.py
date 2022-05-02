#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2021
# ALL RIGHTS RESERVED

# Since the rxn_chemutils dependency requires an environment variable, the
# install_requires variable must be set here instead of setup.cfg.

from setuptools import setup

setup(
    install_requires=[
        'pandas>=0.23.3',  # not installing from setup.cfg
        'tqdm>=4.30.0',  # not installing from setup.cfg
        'jupyterlab>=3.2.0',  # not installing from setup.cfg
        'rxn_chemutils',
        'scikit-learn>=0.23.1',
        'seaborn>=0.11.2',
        'matplotlib>=3.2.2'
    ]
)
