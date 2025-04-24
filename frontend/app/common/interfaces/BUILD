load("@npm//@bazel/concatjs:index.bzl", "ts_library")

package(default_visibility = ["//frontend:internal"])

genrule(
    name = "hlo_proto_ts",
    srcs = ["hlo.jsonpb_decls.d.ts.gz"],
    outs = ["hlo.jsonpb_decls.d.ts"],
    cmd = "gzip -d -c $(SRCS) > $(OUTS)",
    visibility = [
        "//frontend:__subpackages__",
        "//plugin:__subpackages__",
    ],
)

genrule(
    name = "xla_data_proto_ts",
    srcs = ["xla_data.jsonpb_decls.d.ts.gz"],
    outs = ["xla_data.jsonpb_decls.d.ts"],
    cmd = "gzip -d -c $(SRCS) > $(OUTS)",
    visibility = [
        "//frontend:__subpackages__",
        "//plugin:__subpackages__",
    ],
)

genrule(
    name = "memory_profile_proto_ts",
    srcs = ["memory_profile.jsonpb_decls.d.ts.gz"],
    outs = ["memory_profile.jsonpb_decls.d.ts"],
    cmd = "gzip -d -c $(SRCS) > $(OUTS)",
    visibility = [
        "//frontend:__subpackages__",
        "//plugin:__subpackages__",
    ],
)

ts_library(
    name = "source_info_proto_defs",
    srcs = [
        ":source_info.jsonpb_decls.d.ts",
    ],
)

ts_library(
    name = "op_metrics_proto_defs",
    srcs = [
        ":op_metrics.jsonpb_decls.d.ts",
    ],
)

ts_library(
    name = "op_profile_proto_defs",
    srcs = [
        ":op_profile.jsonpb_decls.d.ts",
    ],
)

ts_library(
    name = "xla_proto_defs",
    srcs = [
        ":hlo.jsonpb_decls.d.ts",
        ":xla_data.jsonpb_decls.d.ts",
    ],
)

ts_library(
    name = "memory_profile_proto_defs",
    srcs = [
        ":memory_profile.jsonpb_decls.d.ts",
    ],
)

ts_library(
    name = "memory_viewer_preprocess_proto_defs",
    srcs = [
        ":memory_viewer_preprocess.jsonpb_decls.d.ts",
    ],
)

ts_library(
    name = "interfaces",
    srcs = [
        "buffer_allocation_info.ts",
        "capture_profile.ts",
        "data_table.ts",
        "diagnostics.ts",
        "graph_viewer.ts",
        "heap_object.ts",
        "hosts.ts",
        "navigation_event.ts",
        "roofline_model.ts",
        "summary_info.ts",
        "tool.ts",
        "window.ts",
    ],
    deps = [
        ":memory_profile_proto_defs",
        ":memory_viewer_preprocess_proto_defs",
        ":op_profile_proto_defs",
        ":xla_proto_defs",
        "@npm//@types/google.visualization",
    ],
)

ts_library(
    name = "chart",
    srcs = [
        "chart.ts",
    ],
    deps = [
        ":interfaces",
        "@npm//@types/google.visualization",
    ],
)
