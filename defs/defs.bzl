# Open-source Bazel macros for XProf.
# Stubbed out using copy_to_bin to copy files to the bin directory,
# allowing the parent package to access them for Angular CLI compilation.

load("@aspect_bazel_lib//lib:copy_to_bin.bzl", "copy_to_bin")

def ts_library(name, srcs, assets = [], deps = [], **kwargs):
    # Strip unsupported arguments
    kwargs.pop("allow_warnings", None)
    kwargs.pop("use_angular_plugin", None)
    kwargs.pop("tsconfig", None)
    copy_to_bin(
        name = name,
        srcs = srcs + assets + deps,
        **kwargs
    )

def xprof_ng_module(name, srcs, assets = [], deps = [], **kwargs):
    # Strip unsupported arguments
    kwargs.pop("allow_warnings", None)
    kwargs.pop("use_angular_plugin", None)
    kwargs.pop("tsconfig", None)
    copy_to_bin(
        name = name,
        srcs = srcs + assets + deps,
        **kwargs
    )

def ts_declaration(name, srcs, deps = [], **kwargs):
    copy_to_bin(
        name = name,
        srcs = srcs + deps,
        **kwargs
    )

def sass_binary(name, src, deps = [], **kwargs):
    # Strip arguments that are not supported by copy_to_bin
    kwargs.pop("sass_stack", None)
    kwargs.pop("sourcemap", None)
    copy_to_bin(
        name = name,
        srcs = [src] + deps,
        **kwargs
    )

def sass_library(name, srcs = [], deps = [], **kwargs):
    copy_to_bin(
        name = name,
        srcs = srcs + deps,
        **kwargs
    )

def rollup_bundle(name, deps = [], **kwargs):
    # Strip rollup-specific arguments
    kwargs.pop("config_file", None)
    kwargs.pop("entry_point", None)
    kwargs.pop("format", None)
    kwargs.pop("link_workspace_root", None)
    copy_to_bin(
        name = name,
        srcs = deps,
        **kwargs
    )
