#!/usr/bin/env python

# Copyright 2017 Sidewalk Labs | https://www.apache.org/licenses/LICENSE-2.0

from setuptools import setup

setup(
    name="doppelganger",
    version="0.1.1",
    description='Population synthesis library',
    author='Kat Busch, Kael Greco and contributors',
    author_email='doppelganger@sidewalklabs.com',
    packages=['doppelganger'],
    install_requires=[
        'cvxpy==0.4.8',
        'numpy>=1.11.0',
        'pandas>=0.19.0',
        'pomegranate==0.7.1',
        'requests>=2.0.0',
        'six>=1.10.0',
        'future>=0.16.0',
        'psycopg2==2.7.3.1',
        'enum34',
    ],
    extras_require={
        'tests': [
            'flake8>=3.5.0',
            'mock>=2.0.0',
            'nose>=1.3.4',
            'coveralls>=1.1',
            'pytest',
        ],
    },

)
