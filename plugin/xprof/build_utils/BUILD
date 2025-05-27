load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")

visibility = ["//plugin:internal"]

package(
    default_visibility = visibility,
    licenses = ["notice"],  # Apache 2.0
)

bzl_library(
    name = "strict_default_bzl",
    srcs = ["strict.default.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)

bzl_library(
    name = "pytype_default_bzl",
    srcs = ["pytype.default.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)
