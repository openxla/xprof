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

load("@npm//@bazel/concatjs:index.bzl", "ts_library")

def _convert_paths_to_absolute_impl(ctx):
  """Implementation of convert_paths_to_absolute.

  Used on TypeScript source files to convert their relative import paths
  to absolute import paths.
  """
  outputs = []
  for src in ctx.files.srcs:
    out_file = ctx.actions.declare_file(src.basename)
    outputs.append(out_file)

    ctx.actions.run_shell(
      inputs = [src],
      outputs = [out_file],
      command = """sed "s;from './;from 'org_xprof/%s/;g" '%s' > '%s'""" % (src.dirname, src.path, out_file.path)
    )

  return [DefaultInfo(
      files = depset(
          outputs,
      ),
  )]

convert_paths_to_absolute = rule(
    implementation = _convert_paths_to_absolute_impl,
    attrs = {
        "srcs": attr.label_list(
          doc = "Convert relative import paths to absolute for TypeScript source files.",
          allow_files = [".ts", ".tsx"],
          mandatory = True,
        ),
    },
)

def xprof_ng_module(name, srcs, assets = [], allow_warnings = None, **kwargs):
    """Wrapper for Angular modules for the external BUILD rules"""

    # A hack for ngcc (Angular Ivy) compiler with rules_nodejs's
    # ts_library rule which does not resolve Modules properly
    # if their paths are relative. So we convert all our Module
    # import paths from relative to absolute paths.
    converted_srcs_name = "%s_converted_srcs" % name
    convert_paths_to_absolute(
      name = converted_srcs_name,
      srcs = srcs
    )
    srcs = [":%s" % (converted_srcs_name)]

    # A hack to include @angular/common, @angular/core since the ng_module
    # macro in google3 automatically adds this dep.
    # See: //javascript/angular2/ng_module.bzl
    if "@npm//@angular/common" not in kwargs['deps']:
      kwargs['deps'] += ["@npm//@angular/common"]
    if "@npm//@angular/core" not in kwargs['deps']:
      kwargs['deps'] += ["@npm//@angular/core"]

    ts_library(
        name = name,
        compiler = "//defs:tsc_wrapped_with_angular",
        supports_workers = True,
        use_angular_plugin = True,
        angular_assets = assets,
        srcs = srcs,
        **kwargs
    )
