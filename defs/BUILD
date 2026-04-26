load("@build_bazel_rules_nodejs//:index.bzl", "nodejs_binary")

# In OSS-only, the `xprof_ng_module` rule depends on a
# tsc_wrapped_with_angular nodejs_binary target.
nodejs_binary(
    name = "tsc_wrapped_with_angular",
    data = [
        "@npm//@angular/compiler-cli",
        "@npm//@bazel/concatjs",
        "@npm//@bazel/typescript",
    ],
    entry_point = "@npm//:node_modules/@bazel/concatjs/internal/tsc_wrapped/tsc_wrapped.js",
    visibility = ["//:__subpackages__"],
)

constraint_setting(name = "emscripten_local_routing")

constraint_value(
    name = "emscripten_use_local",
    constraint_setting = ":emscripten_local_routing",
    visibility = ["//visibility:public"],
)

platform(
    name = "emscripten_local_host",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        ":emscripten_use_local",
    ],
    exec_properties = {
        "no-remote-exec": "true",
    },
    visibility = ["//visibility:public"],
)

toolchain(
    name = "cc-toolchain-wasm-local-routed",
    exec_compatible_with = [":emscripten_use_local"],
    target_compatible_with = ["@platforms//cpu:wasm32"],
    toolchain = "@emsdk//emscripten_toolchain:cc-compiler-wasm",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
    visibility = ["//visibility:public"],
)
