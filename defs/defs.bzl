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

"""External-only delegates for various frontend BUILD rules."""

load("@aspect_rules_ts//ts:defs.bzl", "ts_project")

def xprof_ng_module(name, srcs, assets = [], allow_warnings = None, **kwargs):
    """Wrapper for Angular modules for the external BUILD rules"""

    if 'deps' not in kwargs:
      kwargs['deps'] = []
    if "//:node_modules/@angular/common" not in kwargs['deps']:
      kwargs['deps'] += ["//:node_modules/@angular/common"]
    if "//:node_modules/@angular/core" not in kwargs['deps']:
      kwargs['deps'] += ["//:node_modules/@angular/core"]

    ts_project(
        name = name,
        tsconfig = "//:tsconfig",
        declaration = True,
        assets = assets,
        srcs = srcs,
        **kwargs
    )

ts_library = xprof_ng_module
