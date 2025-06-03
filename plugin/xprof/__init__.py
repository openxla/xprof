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
import packaging.version


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
    dist_names = dist_map.get(current_import_name)

    if dist_names:
      if len(dist_names) > 1:
        # Both xprof and xprof-nightly are installed.
        raise RuntimeError(
            "Multiple distributions found for package:"
            f" {current_import_name} ({dist_names})\n"
            "Please uninstall one of the conflicting packages."
        )
      return dist_names[0]

  except (ValueError, IndexError, TypeError, AttributeError):
    pass

  return current_import_name


def _check_for_conflicts(current_package_name):
  """Checks for conflicting legacy packages and raises an error if found."""
  # These are the legacy packages that conflict with ANY new version.
  conflicting_packages = ["tensorboard-plugin-profile", "tbp-nightly"]
  conflict_version_ceiling = "2.20.0"
  for conflicting_pkg in conflicting_packages:
    try:
      installed_version_str = metadata.version(conflicting_pkg)
      installed_version = packaging.version.Version(installed_version_str)
      conflict_version_ceiling_version = packaging.version.Version(
          conflict_version_ceiling
      )
      if installed_version < conflict_version_ceiling_version:
        raise RuntimeError(
            f"Installation Conflict: The package '{current_package_name}'"
            " cannot be used while"
            f" '{conflicting_pkg}=={installed_version_str}' is"
            f" installed.\n\n'{current_package_name}' is entirely backwards-"
            "compatible with Tensorboard. \nTo fix this, please uninstall"
            f" {conflicting_pkg} by running:\n\n  pip uninstall"
            f" {conflicting_pkg}"
        )
    except metadata.PackageNotFoundError:
      # No conflicting package installed.
      continue

_check_for_conflicts(_get_current_package_name())
