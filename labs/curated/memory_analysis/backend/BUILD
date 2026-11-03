load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//xprof:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "hlo_to_memory_analysis",
    srcs = ["hlo_to_memory_analysis.cc"],
    hdrs = ["hlo_to_memory_analysis.h"],
    deps = [
        "//xprof/convert:hlo_proto_to_memory_visualization_utils",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_module_config",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "hlo_to_memory_analysis_test",
    srcs = ["hlo_to_memory_analysis_test.cc"],
    deps = [
        ":hlo_to_memory_analysis",
        "//xprof/utils:tensorflow_utils",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/platform:statusor",
    ],
)
