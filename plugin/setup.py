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

import importlib
import os
from typing import Any, List, Optional

import setuptools

from xprof import version

_PEP639_MIN_SETUPTOOLS_VERSION = '77.0.3'


def _supports_pep639_license() -> bool:
  """Returns whether setuptools supports PEP 639 license_expression.

  Kept intentionally mirrored with tbp_package/setup.py because the two
  setup.py scripts are staged independently for wheel builds.
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




def _load_bdist_wheel_base() -> type[Any] | None:
  """Imports the bdist_wheel command class, preferring setuptools.

  Returns:
    The imported bdist_wheel command class, or None if neither
    setuptools.command.bdist_wheel nor wheel.bdist_wheel can be imported.
  """
  for module_name in ('setuptools.command.bdist_wheel', 'wheel.bdist_wheel'):
    try:
      module = importlib.import_module(module_name)
      return module.bdist_wheel
    except ImportError:
      continue
  return None


def _make_custom_bdist_wheel() -> type[Any] | None:
  """Creates the CustomBdistWheel command class, or None if unavailable."""
  base_cls = _load_bdist_wheel_base()
  if base_cls is None:
    return None

  class _CustomBdistWheel(base_cls):

    def finalize_options(self):
      super().finalize_options()
      self.root_is_pure = False

    def get_tag(self):
      return ('py3', 'none') + super().get_tag()[2:]

  _CustomBdistWheel.__name__ = 'CustomBdistWheel'
  _CustomBdistWheel.__qualname__ = 'CustomBdistWheel'
  return _CustomBdistWheel


CustomBdistWheel = _make_custom_bdist_wheel()  # pylint: disable=invalid-name


def _find_requirements_file() -> str:
  """Locates requirements.in across development and packaging environments."""
  setup_dir = os.path.dirname(os.path.abspath(__file__))
  candidates = [
      os.path.join(setup_dir, 'requirements.in'),
      os.path.join(setup_dir, '..', 'requirements.in'),
  ]

  # Check Bazel runfiles directories
  runfiles_roots = []
  for env_var in ('TEST_SRCDIR', 'PYTHON_RUNFILES', 'RUNFILES_DIR'):
    val = os.environ.get(env_var)
    if val:
      runfiles_roots.append(val)

  runfiles_subpaths = [
      os.path.join('google3', 'third_party', 'xprof', 'requirements.in'),
      os.path.join('org_xprof', 'requirements.in'),
      os.path.join('__main__', 'third_party', 'xprof', 'requirements.in'),
      os.path.join('__main__', 'requirements.in'),
      os.path.join('third_party', 'xprof', 'requirements.in'),
      'requirements.in',
  ]

  for root in runfiles_roots:
    for subpath in runfiles_subpaths:
      candidates.append(os.path.join(root, subpath))

  # Fallback paths relative to CWD
  cwd = os.getcwd()
  candidates.extend([
      os.path.join(cwd, 'requirements.in'),
      os.path.join(cwd, 'third_party', 'xprof', 'requirements.in'),
      os.path.join(cwd, 'google3', 'third_party', 'xprof', 'requirements.in'),
  ])

  for path in candidates:
    if os.path.isfile(path):
      return os.path.abspath(path)

  raise FileNotFoundError(
      'Could not locate requirements.in. Searched candidates: %s' % candidates
  )


def parse_requirements(
    requirements_path: Optional[str] = None,
) -> List[str]:
  """Parses dependencies from requirements.in, stripping comments."""
  if requirements_path is None:
    requirements_path = _find_requirements_file()

  if not os.path.isfile(requirements_path):
    raise FileNotFoundError(
        'Requirements file not found: %s' % requirements_path
    )

  requirements = []
  with open(requirements_path, 'r', encoding='utf-8') as f:
    for raw_line in f:
      line = raw_line.strip()
      if '#' in line:
        line = line.split('#', 1)[0].strip()
      if not line or line.startswith('-'):
        continue
      requirements.append(line)
  return requirements


PROJECT_NAME = 'xprof'
VERSION = version.__version__
REQUIRED_PACKAGES = parse_requirements()
# Dependencies required only to run the test suite. Declared as the `[test]`
# extra so a clean environment can install them via `pip install xprof[test]`;
# without these, tests error out on missing `jax` and `pandas`.
TEST_PACKAGES = [
    'jax',
    'pandas',
]
PACKAGE_DATA = {
    'xprof': [
        'static/**',
        'utils/*.h',
        'convert/profiler_plugin_c_api.so',
        'convert/profiler_plugin_c_api.pyd',
        'convert/profiler_plugin_c_api.dylib',
        'convert/profiler_plugin_c_api.dll',
        'convert/events_db/python/libpywrap_events_db_c_api.so',
        'convert/events_db/python/libpywrap_events_db_c_api.pyd',
        'skills/**',
        'cli/internal/oss/hlo_descriptor_set.pb',
    ],
}
ENTRY_POINTS = {
    'tensorboard_plugins': [
        'profile = xprof.profile_plugin_loader:ProfilePluginLoader',
    ],
    'console_scripts': [
        'xprof = xprof.cli.xprof_cli:main',
        'xprof-install-skills = xprof.cli.tools.install_skills_tool:main',
        'xparity = xprof.xparity.xparity_cli:main',
    ],
}


def get_readme() -> str:
  """Reads and returns the package README file contents."""
  setup_dir = os.path.dirname(os.path.abspath(__file__))
  candidates = [
      os.path.join(setup_dir, 'README.md'),
      os.path.join(setup_dir, '..', 'README.md'),
      os.path.join(setup_dir, 'README.rst'),
      os.path.join(os.getcwd(), 'README.md'),
      'README.md',
  ]
  for path in candidates:
    if os.path.isfile(path):
      with open(path, 'r', encoding='utf-8') as f:
        return f.read()
  return 'XProf Profiler Plugin'


cmdclass = {}
if CustomBdistWheel:
  cmdclass['bdist_wheel'] = CustomBdistWheel


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
      'version': VERSION,
      'description': 'XProf Profiler Plugin',
      'long_description': get_readme(),
      'long_description_content_type': 'text/markdown',
      'author': 'Google Inc.',
      'author_email': 'packages@tensorflow.org',
      'url': 'https://github.com/openxla/xprof',
      'packages': (
          setuptools.find_packages()
          + setuptools.find_namespace_packages(
              include=['xprof.*'],
              exclude=['xprof.static'],
          )
      ),
      'package_data': PACKAGE_DATA,
      'entry_points': ENTRY_POINTS,
      'has_ext_modules': lambda: True,
      'cmdclass': cmdclass,
      'python_requires': '>= 3.10',
      'install_requires': REQUIRED_PACKAGES,
      'extras_require': {'test': TEST_PACKAGES},
      # PyPI package information.
      'classifiers': classifiers,
      'keywords': 'jax pytorch xla tensorflow tensorboard xprof profile plugin',
      **license_kwargs,
  }


if __name__ == '__main__':
  setuptools.setup(**get_setup_kwargs())
