#!/usr/bin/env python

# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from setuptools import setup


def get_requirements():
    """Return the components defined in requirements.txt."""
    with open('requirements.txt') as fin:
        return [l.split()[0].split('#', 1)[0].strip() for l in fin
                if l and not l.startswith('git')]


setup(
    name="doppelganger",
    version="0.1.4",
    description='Population synthesis library',
    author='Kat Busch, Kael Greco and contributors',
    author_email='doppelganger@sidewalklabs.com',
    packages=['doppelganger'],
    install_requires=get_requirements(),
    extras_require={
        'tests': [
            'flake8>=3.5.0',
            'mock>=2.0.0',
            'coveralls>=1.1',
            'pytest',
            'pytest-cov',
        ],
    },

)
