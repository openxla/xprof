# LLO static-schedule analyses for xprof_cli (shared 1P + 3P OSS library).
#
# Reads the full LloModuleProto embedded in an XSpace metadata plane using
# llo_lite.proto and derives region trees, per-bundle static utilization,
# register pressure, spills/allocations, opcode stats, BDI stalls, target
# arguments, and exact source-line <-> bundle mappings as JSON.

load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = [
    "//plugin/xprof/cli:__subpackages__",
    # _pywrap_profiler_plugin falls back to this pure-Python implementation
    # when the binary is built without embedded features.
    "@org_xprof//xprof/pywrap:__pkg__",
])

pytype_strict_library(
    name = "xspace_llo_reader",
    srcs = ["xspace_llo_reader.py"],
    deps = [
        "//plugin/xprof/protobuf:llo_lite_proto_py",
        "@absl_py//absl/logging",
        "@pypi//protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_py_proto",
    ],
)

pytype_strict_library(
    name = "llo_region_tree",
    srcs = ["llo_region_tree.py"],
    deps = [
        "//plugin/xprof/protobuf:llo_lite_proto_py",
    ],
)

pytype_strict_library(
    name = "llo_bundle_utilization",
    srcs = ["llo_bundle_utilization.py"],
    deps = [
        "//plugin/xprof/protobuf:llo_lite_proto_py",
    ],
)

pytype_strict_library(
    name = "llo_source_map",
    srcs = ["llo_source_map.py"],
    deps = [
        ":llo_region_tree",
        "//plugin/xprof/protobuf:llo_lite_proto_py",
    ],
)

pytype_strict_library(
    name = "register_pressure",
    srcs = ["register_pressure.py"],
    deps = [
        ":llo_region_tree",
        "//plugin/xprof/protobuf:llo_lite_proto_py",
    ],
)

pytype_strict_library(
    name = "llo_allocations",
    srcs = ["llo_allocations.py"],
    deps = [
        ":llo_region_tree",
        "//plugin/xprof/protobuf:llo_lite_proto_py",
    ],
)

pytype_strict_library(
    name = "llo_opcode_stats",
    srcs = ["llo_opcode_stats.py"],
    deps = [
        ":llo_region_tree",
        "//plugin/xprof/protobuf:llo_lite_proto_py",
    ],
)

pytype_strict_library(
    name = "bdi_stalls",
    srcs = ["bdi_stalls.py"],
    deps = [
        ":llo_region_tree",
        "//plugin/xprof/protobuf:llo_lite_proto_py",
    ],
)

pytype_strict_library(
    name = "target_info",
    srcs = ["target_info.py"],
    deps = [
        "//plugin/xprof/protobuf:llo_lite_proto_py",
    ],
)

pytype_strict_library(
    name = "llo_test_fixtures",
    testonly = True,
    srcs = ["llo_test_fixtures.py"],
    deps = [
        "//plugin/xprof/protobuf:llo_lite_proto_py",
        "@tsl//tsl/profiler/protobuf:xplane_proto_py_proto",
    ],
)

pytype_strict_contrib_test(
    name = "llo_static_analysis_test",
    srcs = ["llo_static_analysis_test.py"],
    deps = [
        ":llo_bundle_utilization",
        ":llo_region_tree",
        ":llo_source_map",
        ":xspace_llo_reader",
        "//plugin/xprof/protobuf:llo_lite_proto_py",
        "@absl_py//absl/testing:absltest",
        "@tsl//tsl/profiler/protobuf:xplane_proto_py_proto",
    ],
)

pytype_strict_contrib_test(
    name = "llo_analyses_test",
    srcs = ["llo_analyses_test.py"],
    deps = [
        ":bdi_stalls",
        ":llo_allocations",
        ":llo_opcode_stats",
        ":register_pressure",
        ":target_info",
        "//plugin/xprof/protobuf:llo_lite_proto_py",
        "@absl_py//absl/testing:absltest",
    ],
)
