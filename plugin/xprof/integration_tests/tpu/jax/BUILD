load("@org_xprof//plugin/xprof/build_utils:profiler_test.bzl", "profiler_test")

profiler_test(
    name = "hlo_metadata_test",
    srcs = ["hlo_metadata_test.py"],
    main = "hlo_metadata_test.py",
    platforms = [
        "v2",
        "v3",
        "v4",
        "v6e",
    ],
    tags = [
        "cloudtpu-1vm-profiler",
        "requires-net:external",
    ],
    deps = [
        "//third_party/py/jax",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
        "@org_xprof//plugin/xprof/convert:raw_to_tool_data",
    ],
)

profiler_test(
    name = "jax_profiler_test",
    srcs = ["jax_profiler_test.py"],
    main = "jax_profiler_test.py",
    platforms = [
        "v4",
        "v6e",
    ],
    tags = [
        "cloudtpu-1vm-profiler",
        "requires-net:external",
    ],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
    ],
)
