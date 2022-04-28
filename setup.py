#!/usr/bin/env python
# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2021
# ALL RIGHTS RESERVED

# Since the rxn_chemutils dependency requires an environment variable, the
# install_requires variable must be set here instead of setup.cfg.

import os
from setuptools import setup

setup(
    install_requires=[
        'pandas>=0.23.3',  # not installing from setup.cfg
        'tqdm>=4.30.0',  # not installing from setup.cfg
        'jupyterlab>=3.2.0',  # not installing from setup.cfg
        'rxn_chemutils'
#       DEPRECATED
#        ' @ git+https://{}@github.ibm.com/rxn/rxn_chemutils@0.3.11'.format(os.environ['GHE_TOKEN']),
    ]
)
