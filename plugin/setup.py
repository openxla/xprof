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

import os
from typing import List, Optional

import setuptools

from xprof import version

try:
  from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel  # pylint: disable=g-import-not-at-top

  class CustomBdistWheel(_bdist_wheel):

    def finalize_options(self):
      _bdist_wheel.finalize_options(self)
      self.root_is_pure = False

    def get_tag(self):
      return ('py3', 'none') + _bdist_wheel.get_tag(self)[2:]

except ImportError:
  CustomBdistWheel = None  # pylint: disable=invalid-name


def _find_requirements_file() -> str:
  """Locates the requirements.in file across development and packaging environments."""
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
  """Parses dependencies from a requirements.in file, stripping comments and blanks."""
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


if __name__ == '__main__':
  setuptools.setup(
      name=PROJECT_NAME,
      version=VERSION,
      description='XProf Profiler Plugin',
      long_description=get_readme(),
      long_description_content_type='text/markdown',
      author='Google Inc.',
      author_email='packages@tensorflow.org',
      url='https://github.com/openxla/xprof',
      packages=setuptools.find_packages()
      + setuptools.find_namespace_packages(
          include=['xprof.*'],
          exclude=['xprof.static'],
      ),
      package_data={
          'xprof': [
              'static/**',
              'utils/*.h',
              'convert/profiler_plugin_c_api.so',
              'convert/profiler_plugin_c_api.pyd',
              'convert/profiler_plugin_c_api.dylib',
              'convert/profiler_plugin_c_api.dll',
          ],
      },
      entry_points={
          'tensorboard_plugins': [
              'profile = xprof.profile_plugin_loader:ProfilePluginLoader',
          ],
          'console_scripts': [
              'xprof = xprof.cli.xprof_cli:main',
          ],
      },
      cmdclass=cmdclass,
      python_requires='>= 3.10',
      install_requires=REQUIRED_PACKAGES,
      tests_require=REQUIRED_PACKAGES,
      # PyPI package information.
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Software Development :: Libraries',
      ],
      license='Apache 2.0',
      keywords='jax pytorch xla tensorflow tensorboard xprof profile plugin',
  )
