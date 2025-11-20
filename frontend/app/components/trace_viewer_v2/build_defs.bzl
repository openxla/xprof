"""
This module contains global variables and configurations used in the BUILD files.
"""

load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")
load("//third_party/bazel_rules/rules_wasm/wasm:defs.bzl", "wasm_web_test")

# Tags to apply to targets that are part of the WASM build and cannot be built with standard tools.
# This includes libraries, binaries, and tests that will be run in a WASM environment.
WASM_TAGS = [
    "manual",
    "nobuilder",
    "notap",
    "requires_wasm_config",
]

TEST_LINKOPTS = ["-sASYNCIFY=1", "-sNO_EXIT_RUNTIME=1", "-sUSE_PTHREADS=1"]

def wasm_cc_test(name, srcs, deps = [], copts = [], linkopts = TEST_LINKOPTS, **kwargs):
    """Generates a cc_test and a wasm_web_test target.

    Args:
      name: The name of the test.
      srcs: The source files for the test.
      deps: The dependencies for the test.
      copts: The compilation options for the test.
      linkopts: The link options for the test.
      **kwargs: Additional arguments to pass to the cc_test.
    """
    cc_test(
        name = name + "_cc",
        srcs = srcs,
        copts = copts,
        linkopts = linkopts,
        tags = WASM_TAGS,
        deps = deps,
        **kwargs
    )

    wasm_web_test(
        name = name,
        cc_target = ":" + name + "_cc",
    )

def wasm_cc_library(name, **kwargs):
    """Generates a cc_library target with WASM_TAGS.

    Args:
      name: The name of the library.
      **kwargs: Additional arguments to pass to the cc_library.
    """
    cc_library(
        name = name,
        tags = WASM_TAGS,
        **kwargs
    )
