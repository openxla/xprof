# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

from tensorboard_plugin_profile import version


PROJECT_NAME = 'tensorboard_plugin_profile'
VERSION = version.__version__
REQUIRED_PACKAGES = [
    'gviz_api >= 1.9.0',
    'protobuf >= 3.19.6,<5.0.0dev',
    'setuptools >= 41.0.0',
    'six >= 1.10.0',
    'werkzeug >= 0.11.15',
    'etils[epath] >= 1.0.0',
    'cheroot >= 10.0.1',
    'fsspec >= 2024.10.0',
]


def get_readme():
  with open('README.md') as f:
    return f.read()


setuptools.setup(
    name=PROJECT_NAME,
    version=VERSION,
    description='Profile Tensorboard Plugin',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    url='https://github.com/openxla/xprof',
    packages=setuptools.find_packages()
    + setuptools.find_namespace_packages(
        include=['tensorboard_plugin_profile.*'],
        exclude=['tensorboard_plugin_profile.static'],
    ),
    package_data={
        'tensorboard_plugin_profile': ['static/**'],
        '': ['_pywrap_profiler_plugin.so'],
    },
    entry_points={
        'tensorboard_plugins': [
            (
                'profile ='
                ' tensorboard_plugin_profile.profile_plugin_loader:ProfilePluginLoader'
            ),
        ],
        'console_scripts': [
            'xprof = tensorboard_plugin_profile.server:main',
        ],
    },
    python_requires='>= 2.7, != 3.0.*, != 3.1.*',
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES,
    # PyPI package information.
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
    license='Apache 2.0',
    keywords='jax pytorch xla tensorflow tensorboard xprof profile plugin',
)
