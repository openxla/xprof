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

load("@npm//@bazel/concatjs:index.bzl", _concatjs_ts_library = "ts_library")

def ts_library(name, srcs, assets = [], allow_warnings = None, **kwargs):
    """Wrapper for ts_library in external BUILD rules"""
    angular_deps = [
        "@npm//@angular/animations",
        "@npm//@angular/cdk",
        "@npm//@angular/common",
        "@npm//@angular/compiler",
        "@npm//@angular/core",
        "@npm//@angular/forms",
        "@npm//@angular/localize",
        "@npm//@angular/material",
        "@npm//@angular/platform-browser",
        "@npm//@angular/platform-browser-dynamic",
        "@npm//@angular/router",
        "@npm//@types/chai",
        "@npm//@types/emscripten",
        "@npm//@types/google.visualization",
        "@npm//@types/jasmine",
        "@npm//@types/node",
        "@npm//@types/sinon",
        "@npm//rxjs",
        "@npm//tslib",
    ]
    if "deps" not in kwargs:
      kwargs["deps"] = []
    for dep in angular_deps:
      if dep not in kwargs["deps"]:
        kwargs["deps"].append(dep)

    if "tsconfig" not in kwargs:
      kwargs["tsconfig"] = "//:tsconfig.json"

    _concatjs_ts_library(
        name = name,
        supports_workers = True,
        prodmode_target = "esnext",
        devmode_target = "esnext",
        devmode_module = "esnext",
        use_angular_plugin = False,
        angular_assets = assets,
        srcs = srcs,
        **kwargs
    )

def xprof_ng_module(name, srcs, assets = [], allow_warnings = None, **kwargs):
    """Wrapper for Angular modules for the external BUILD rules"""
    ts_library(
        name = name,
        assets = assets,
        srcs = srcs,
        allow_warnings = allow_warnings,
        **kwargs
    )
