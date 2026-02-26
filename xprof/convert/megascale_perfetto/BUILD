# Package for generating Megascale Perfetto traces.

# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "xprof_trace",
    srcs = [],
    hdrs = ["xprof_trace.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@xla//xla/tsl/lib/gtl:int_type",
    ],
)

cc_library(
    name = "xspace_loader",
    srcs = ["xspace_loader.cc"],
    hdrs = ["xspace_loader.h"],
    deps = [
        ":xprof_trace",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_googlesource_code_re2//:re2",
        "@org_xprof//xprof/convert:file_utils",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "perfetto_writer",
    srcs = ["perfetto_writer.cc"],
    hdrs = ["perfetto_writer.h"],
    deps = [
        ":xprof_trace",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_protobuf//:protobuf",
        "@perfetto//:trace_cc_proto",
    ],
)

cc_library(
    name = "trace_processor",
    srcs = ["trace_processor.cc"],
    hdrs = ["trace_processor.h"],
    deps = [
        ":xprof_trace",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_googlesource_code_re2//:re2",
        "@xla//xla/tsl/lib/gtl:map_util",
    ],
)

cc_test(
    name = "xspace_loader_test",
    srcs = ["xspace_loader_test.cc"],
    deps = [
        ":xprof_trace",
        ":xspace_loader",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
    ],
)

cc_test(
    name = "trace_processor_test",
    srcs = ["trace_processor_test.cc"],
    deps = [
        ":trace_processor",
        ":xprof_trace",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "perfetto_writer_test",
    srcs = ["perfetto_writer_test.cc"],
    deps = [
        ":perfetto_writer",
        ":xprof_trace",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@perfetto//:trace_cc_proto",
    ],
)
