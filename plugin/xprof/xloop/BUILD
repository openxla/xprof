# XLoop Engine: microbenchmarking for JAX functions and Pallas kernels.
#
# Shared 1P + 3P library. Backends that depend on Google-internal services are
# fenced with copybara strip markers rather than living in a separate tree, so
# both sides build from one definition.

load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_visibility = [
        "//plugin:internal",
        "@org_xprof//plugin:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

pytype_strict_library(
    name = "input_utils",
    srcs = ["input_utils.py"],
    deps = [
        "//plugin/xprof/protobuf:xloop_item_proto_py",
        "@pypi//protobuf",
    ],
)

pytype_strict_library(
    name = "chip_broker",
    srcs = ["chip_broker.py"],
)

# Requires JAX. Kept a separate target from the rest of the engine so that
# `xprof` itself, and the CLI registry, import without a JAX install; the
# open-source wheel declares JAX under the `xloop` extra rather than as a
# runtime dependency.
pytype_strict_library(
    name = "eval_jax_lib",
    srcs = ["eval_jax_lib.py"],
)

pytype_strict_library(
    name = "trace_timing_lib",
    srcs = ["trace_timing_lib.py"],
    deps = [
        "//plugin/xprof:profile_data",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_contrib_test(
    name = "trace_timing_lib_test",
    srcs = ["trace_timing_lib_test.py"],
    deps = [
        ":trace_timing_lib",
        "@absl_py//absl/testing:absltest",
        "@tsl//tsl/profiler/protobuf:xplane_proto_py_proto",
    ],
)

pytype_strict_contrib_test(
    name = "calibration_test",
    srcs = ["calibration_test.py"],
    data = ["testdata/dispatch_floor_calibration.jsonl"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":chip_broker",
        ":input_utils",
        ":summarize_results_lib",
        ":xloop_lib",
        "//plugin/xprof/protobuf:xloop_item_proto_py",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "summarize_results_lib",
    srcs = ["summarize_results_lib.py"],
    deps = [
        ":input_utils",
        "//plugin/xprof/protobuf:xloop_item_proto_py",
    ],
)

# Import-safe: no module-level flags, and :eval_jax_lib is imported at call
# time so a JAX-free install can still load the CLI registry.
pytype_strict_library(
    name = "xloop_worker_lib",
    srcs = ["xloop_worker_lib.py"],
    deps = [
        ":eval_jax_lib",
        ":input_utils",
        ":trace_timing_lib",
        "//plugin/xprof/protobuf:xloop_item_proto_py",
    ],
)

pytype_strict_binary(
    name = "xloop_worker",
    srcs = ["xloop_worker_main.py"],
    main = "xloop_worker_main.py",
    visibility = ["//visibility:public"],
    deps = [
        ":input_utils",
        ":xloop_worker_lib",
        "@absl_py//absl/flags",
        "@pypi//absl_py",
    ],
)

pytype_strict_binary(
    name = "xloop",
    srcs = ["xloop_cli.py"],
    main = "xloop_cli.py",
    visibility = ["//visibility:public"],
    deps = [
        ":xloop_cli",
        "@absl_py//absl/flags",
        "@pypi//absl_py",
        "@pypi//fire",
    ],
)

pytype_strict_library(
    name = "xloop_lib",
    srcs = ["xloop_lib.py"],
    deps = [
        ":chip_broker",
        ":input_utils",
        ":summarize_results_lib",
        ":xloop_worker_lib",
        "//plugin/xprof/protobuf:xloop_item_proto_py",
    ],
)

pytype_strict_library(
    name = "xloop_cli",
    srcs = ["xloop_cli.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":chip_broker",
        ":input_utils",
        ":summarize_results_lib",
        ":xloop_lib",
        "//plugin/xprof/protobuf:xloop_item_proto_py",
        "@absl_py//absl/flags",
        "@pypi//absl_py",
        "@pypi//fire",
    ],
)

pytype_strict_contrib_test(
    name = "input_utils_test",
    srcs = ["input_utils_test.py"],
    deps = [
        ":input_utils",
        "//plugin/xprof/protobuf:xloop_item_proto_py",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "chip_broker_test",
    srcs = ["chip_broker_test.py"],
    deps = [
        ":chip_broker",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "eval_jax_lib_test",
    srcs = ["eval_jax_lib_test.py"],
    deps = [
        ":eval_jax_lib",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "summarize_results_lib_test",
    srcs = ["summarize_results_lib_test.py"],
    deps = [
        ":summarize_results_lib",
        "//plugin/xprof/protobuf:xloop_item_proto_py",
        "@absl_py//absl/testing:absltest",
    ],
)

# End-to-end: drives real kernels through the worker on CPU.
pytype_strict_contrib_test(
    name = "xloop_lib_test",
    srcs = ["xloop_lib_test.py"],
    data = [":xloop_worker"],
    shard_count = 2,
    deps = [
        ":input_utils",
        ":summarize_results_lib",
        ":xloop_cli",
        ":xloop_lib",
        "//plugin/xprof/protobuf:xloop_item_proto_py",
        "@absl_py//absl/testing:absltest",
    ],
)
