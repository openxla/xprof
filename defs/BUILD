load("@build_bazel_rules_nodejs//:index.bzl", "nodejs_binary")

# In OSS-only, the `xprof_ng_module` rule depends on a
# tsc_wrapped_with_angular nodejs_binary target.
nodejs_binary(
    name = "tsc_wrapped_with_angular",
    data = [
        "//:node_modules/@angular/compiler-cli",
        "//:node_modules/@bazel/concatjs",
        "//:node_modules/@bazel/typescript",
    ],
    entry_point = "//:node_modules/@bazel/concatjs/internal/tsc_wrapped/tsc_wrapped.js",
    visibility = ["//:__subpackages__"],
)
