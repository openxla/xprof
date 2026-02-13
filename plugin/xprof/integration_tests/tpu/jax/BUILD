load("@org_xprof//plugin/xprof/build_utils:profiler_test.bzl", "profiler_test")

package(
    default_visibility = ["//plugin:internal"],
    licenses = ["notice"],
)

profiler_test(
    name = "jax_profiler_test",
    srcs = ["jax_profiler_test.py"],
    main = "jax_profiler_test.py",
    platforms = ["v4"],
    tags = [
        "requires-net:external",
    ],
    deps = [
        "//learning/brain/research/jax:tpu_support",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@pytest//pytest",
    ],
)
