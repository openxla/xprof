# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__",
])

cc_library(
    name = "colors",
    srcs = ["color_generator.cc"],
    hdrs = [
        "color_generator.h",
        "colors.h",
    ],
    deps = [
        "//third_party/dear_imgui",
        "//util/hash:highway_fingerprint",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "color_generator_test",
    srcs = ["color_generator_test.cc"],
    deps = [
        ":colors",
        "//testing/production_stub/public:testvalue",
        "//third_party/dear_imgui",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
