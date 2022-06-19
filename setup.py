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
        'pandas>=0.23.3',
        'tqdm>=4.30.0',
        'jupyterlab>=3.2.0',
        'rxn-utils',
        'rxn-chem-utils',
        'rxn-opennmt-py>=1.0.3',
        'scikit-learn>=0.23.1',
        'seaborn>=0.11.2',
        'matplotlib>=3.2.2'
    ]
)
