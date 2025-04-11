# Copyright 2025 The Xprof Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Passthrough setup.py pointing to xprof."""

import os

import setuptools

PROJECT_NAME = 'xprof'
VERSION = '0.0.0'


def get_long_description():
  with open(
      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md'),
      encoding='utf8',
  ) as fp:
    return fp.read()


setuptools.setup(
    name=PROJECT_NAME,
    description='XProf Tensorboard Plugin',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    version=VERSION,
    install_requires=[f'tensorboard_plugin_profile=={VERSION}'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    packages=setuptools.find_packages()
    + setuptools.find_namespace_packages(
        include=['xprof.*'],
    ),
    python_requires='>= 2.7, != 3.0.*, != 3.1.*',
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    url='https://github.com/openxla/xprof',
    license='Apache 2.0',
    keywords='jax pytorch xla tensorflow tensorboard xprof profile plugin',
)
