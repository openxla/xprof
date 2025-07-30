# Copyright 2025 The XProf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Entry point for the TensorBoard plugin package for XProf.

Public submodules:
  profile_plugin: The TensorBoard plugin integration.
  profile_plugin_loader: TensorBoard's entrypoint for the plugin.
  server: Standalone server entrypoint.
  version: The version of the plugin.
"""

from importlib import metadata
import warnings


def _get_current_package_name():
  """Discovers the distribution package name (e.g., 'xprof-nightly').
  """
  # __package__ should be 'xprof'
  current_import_name = __package__

  try:
    # packages_distributions() returns a mapping like:
    # {'xprof': ['xprof-nightly'], 'numpy': ['numpy']}
    dist_map = metadata.packages_distributions()

    # Look up our import name to find the list of distributions that provide it.
    # In a standard environment, this list will have one item.
    dist_names = set(dist_map.get(current_import_name))

    if dist_names:
      if len(dist_names) > 1:
        warnings.warn(
            f"Multiple distributions found for package '{current_import_name}':"
            f" {dist_names}. Please uninstall one of them.",
            UserWarning,
        )
      return dist_names[0]

  except (ValueError, IndexError, TypeError, AttributeError):
    pass

  return current_import_name


def _check_for_conflicts():
  """Checks for conflicting legacy packages and raises an error if any are found."""
  current_package_name = _get_current_package_name()

  conflicting_packages = ["tensorboard-plugin-profile", "tbp-nightly"]

  for conflicting_pkg in conflicting_packages:
    try:
      installed_version = metadata.version(conflicting_pkg)

      raise RuntimeError(
          f"Installation Conflict: The package '{current_package_name}'"
          f" cannot be used while '{conflicting_pkg}=={installed_version}' is"
          " installed.\n\nTo fix this, please uninstall the conflicting package"
          f" by running:\n\n  pip uninstall {conflicting_pkg}"
      )
    except metadata.PackageNotFoundError:
      continue


_check_for_conflicts()
