# Copyright 2025 The XProf Authors. All Rights Reserved.
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

import importlib
import os
from typing import Any

import setuptools

PROJECT_NAME = 'tensorboard_plugin_profile'
VERSION = '0.0.0'

_PEP639_MIN_SETUPTOOLS_VERSION = '77.0.3'


def _supports_pep639_license() -> bool:
  """Returns whether setuptools supports PEP 639 license_expression.

  Kept intentionally mirrored with plugin/setup.py because the two setup.py
  scripts are staged independently for wheel builds.
  """
  # Setuptools >= 77.0.3 requires packaging.licenses (packaging >= 24.2) to
  # finalize PEP 639 metadata; fall back early if an older site-packages
  # packaging shadows the vendored copy.
  try:
    importlib.import_module('packaging.licenses')
    packaging_version = importlib.import_module('packaging.version')
  except ImportError:
    return False
  # Check the parsed PEP 440 version first: earlier 77.0.x releases already
  # exposed metadata.license_expression before PEP 639 support stabilized in
  # 77.0.3, and this avoids instantiating Distribution on standard releases.
  # Reserve attribute probing for unversioned or vendored builds that report
  # '0.dev0+unknown' or non-PEP-440 version strings.
  try:
    parsed_version = packaging_version.Version(setuptools.__version__)
  except packaging_version.InvalidVersion:
    parsed_version = None
  if parsed_version is not None and (
      parsed_version.epoch > 0 or parsed_version.major > 0
  ):
    return parsed_version >= packaging_version.Version(
        _PEP639_MIN_SETUPTOOLS_VERSION
    )
  return hasattr(setuptools.Distribution().metadata, 'license_expression')


def _get_license_metadata() -> tuple[tuple[str, ...], dict[str, str]]:
  """Returns (license_classifiers, license_kwargs) for setuptools.setup()."""
  if _supports_pep639_license():
    return (), {'license_expression': 'Apache-2.0'}
  return (
      ('License :: OSI Approved :: Apache Software License',),
      {'license': 'Apache 2.0'},
  )




def get_long_description() -> str:
  """Reads and returns the package README file contents."""
  setup_dir = os.path.dirname(os.path.abspath(__file__))
  for path in (
      os.path.join(setup_dir, 'README.md'),
      os.path.join(setup_dir, '..', 'README.md'),
  ):
    if os.path.isfile(path):
      with open(path, encoding='utf8') as fp:
        return fp.read()
  return 'XProf Profiler Plugin'


def get_setup_kwargs() -> dict[str, Any]:
  """Returns keyword arguments for setuptools.setup()."""
  license_classifiers, license_kwargs = _get_license_metadata()
  classifiers = [
      'Intended Audience :: Developers',
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      *license_classifiers,
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Topic :: Software Development :: Libraries :: Python Modules',
      'Topic :: Software Development :: Libraries',
  ]
  return {
      'name': PROJECT_NAME,
      'description': 'XProf Profiler Plugin',
      'long_description': get_long_description(),
      'long_description_content_type': 'text/markdown',
      'version': VERSION,
      'install_requires': [f'xprof=={VERSION}'],
      'classifiers': classifiers,
      'packages': (
          setuptools.find_packages()
          + setuptools.find_namespace_packages(
              include=['xprof.*'],
          )
      ),
      'python_requires': '>= 3.10',
      'author': 'Google Inc.',
      'author_email': 'packages@tensorflow.org',
      'url': 'https://github.com/openxla/xprof',
      'keywords': 'jax pytorch xla tensorflow tensorboard xprof profile plugin',
      **license_kwargs,
  }


if __name__ == '__main__':
  setuptools.setup(**get_setup_kwargs())
