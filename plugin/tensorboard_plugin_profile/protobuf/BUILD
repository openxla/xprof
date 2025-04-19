# Description:
#  Protobufs used by TensorBoard profile plugin.

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@local_xla//xla/tsl/platform:build_config.bzl", xprof_proto_library = "tf_proto_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# copybara:uncomment_begin(google-only)
# package_group(
#     name = "friends",
#     includes = [
#         "@org_tensorflow//tensorflow/core/profiler:friends",
#     ],
#     packages = [
#         "@org_tensorflow//tensorflow/core/profiler/...",
#     ],
# )
# copybara:uncomment_end

xprof_proto_library(
    name = "dcn_collective_info_proto",
    srcs = ["dcn_collective_info.proto"],
)

xprof_proto_library(
    name = "dcn_slack_analysis_proto",
    srcs = ["dcn_slack_analysis.proto"],
)

xprof_proto_library(
    name = "diagnostics_proto",
    srcs = ["diagnostics.proto"],
)

xprof_proto_library(
    name = "hardware_types_proto",
    srcs = ["hardware_types.proto"],
)

xprof_proto_library(
    name = "hlo_stats_proto",
    srcs = ["hlo_stats.proto"],
    protodeps = [":source_info_proto"],
)

xprof_proto_library(
    name = "inference_stats_proto",
    srcs = ["inference_stats.proto"],
)

xprof_proto_library(
    name = "input_pipeline_proto",
    srcs = ["input_pipeline.proto"],
    protodeps = [":diagnostics_proto"],
)

xprof_proto_library(
    name = "kernel_stats_proto",
    srcs = ["kernel_stats.proto"],
)

xprof_proto_library(
    name = "memory_profile_proto",
    srcs = ["memory_profile.proto"],
)

xprof_proto_library(
    name = "memory_viewer_preprocess_proto",
    srcs = ["memory_viewer_preprocess.proto"],
)

xprof_proto_library(
    name = "source_info_proto",
    srcs = ["source_info.proto"],
)

xprof_proto_library(
    name = "op_metrics_proto",
    srcs = ["op_metrics.proto"],
    protodeps = [":source_info_proto"],
)

xprof_proto_library(
    name = "op_profile_proto",
    srcs = ["op_profile.proto"],
    protodeps = [":source_info_proto"],
)

xprof_proto_library(
    name = "op_stats_proto",
    srcs = ["op_stats.proto"],
    protodeps = [
        ":hardware_types_proto",
        ":op_metrics_proto",
        ":power_metrics_proto",
        ":steps_db_proto",
        ":tf_function_proto",
        ":topology_proto",
        ":diagnostics_proto",
        ":kernel_stats_proto",
    ],
)

xprof_proto_library(
    name = "overview_page_proto",
    srcs = ["overview_page.proto"],
    protodeps = [
        ":diagnostics_proto",
        ":input_pipeline_proto",
        ":power_metrics_proto",
    ],
)

xprof_proto_library(
    name = "pod_stats_proto",
    srcs = ["pod_stats.proto"],
    protodeps = [
        ":diagnostics_proto",
    ],
)

xprof_proto_library(
    name = "pod_viewer_proto",
    srcs = ["pod_viewer.proto"],
    protodeps = [
        ":diagnostics_proto",
        ":pod_stats_proto",
        ":topology_proto",
    ],
)

xprof_proto_library(
    name = "power_metrics_proto",
    srcs = ["power_metrics.proto"],
)

xprof_proto_library(
    name = "roofline_model_proto",
    srcs = ["roofline_model.proto"],
    protodeps = [
        ":diagnostics_proto",
        ":source_info_proto",
    ],
)

xprof_proto_library(
    name = "steps_db_proto",
    srcs = ["steps_db.proto"],
    protodeps = [":op_metrics_proto"],
)

xprof_proto_library(
    name = "task_proto",
    srcs = ["task.proto"],
)

xprof_proto_library(
    name = "tensorflow_datatypes_proto",
    srcs = ["tensorflow_datatypes.proto"],
)

xprof_proto_library(
    name = "tf_data_stats_proto",
    srcs = ["tf_data_stats.proto"],
)

xprof_proto_library(
    name = "tf_function_proto",
    srcs = ["tf_function.proto"],
)

xprof_proto_library(
    name = "tf_stats_proto",
    srcs = ["tf_stats.proto"],
)

xprof_proto_library(
    name = "tfstreamz_proto",
    srcs = ["tfstreamz.proto"],
)

xprof_proto_library(
    name = "topology_proto",
    srcs = ["topology.proto"],
)

xprof_proto_library(
    name = "tpu_input_pipeline_proto",
    srcs = ["tpu_input_pipeline.proto"],
    protodeps = [":input_pipeline_proto"],
)

xprof_proto_library(
    name = "trace_events_proto",
    srcs = ["trace_events.proto"],
    protodeps = [":task_proto"],
)

xprof_proto_library(
    name = "trace_events_old_proto",
    srcs = ["trace_events_old.proto"],
    protodeps = [":task_proto"],
)

xprof_proto_library(
    name = "trace_events_raw_proto",
    srcs = ["trace_events_raw.proto"],
)

xprof_proto_library(
    name = "protos_all",
    create_go_proto = False,
    create_java_proto = False,
    protodeps = [
        ":dcn_collective_info_proto",
        ":dcn_slack_analysis_proto",
        ":diagnostics_proto",
        ":hardware_types_proto",
        ":hlo_stats_proto",
        ":inference_stats_proto",
        ":input_pipeline_proto",
        ":kernel_stats_proto",
        ":memory_profile_proto",
        ":memory_viewer_preprocess_proto",
        ":source_info_proto",
        ":op_metrics_proto",
        ":op_profile_proto",
        ":op_stats_proto",
        ":overview_page_proto",
        ":pod_stats_proto",
        ":pod_viewer_proto",
        ":power_metrics_proto",
        ":roofline_model_proto",
        ":steps_db_proto",
        ":task_proto",
        ":tf_data_stats_proto",
        ":tf_function_proto",
        ":tf_stats_proto",
        ":tfstreamz_proto",
        ":topology_proto",
        ":tpu_input_pipeline_proto",
        ":trace_events_proto",
        ":trace_events_old_proto",
        ":trace_events_raw_proto",
    ],
)

exports_files(
    srcs = [
        "dcn_collective_info.proto",
        "dcn_slack_analysis.proto",
        "diagnostics.proto",
        "hardware_types.proto",
        "hlo_stats.proto",
        "inference_stats.proto",
        "input_pipeline.proto",
        "kernel_stats.proto",
        "memory_profile.proto",
        "memory_viewer_preprocess.proto",
        "op_metrics.proto",
        "op_profile.proto",
        "op_stats.proto",
        "overview_page.proto",
        "pod_stats.proto",
        "pod_viewer.proto",
        "power_metrics.proto",
        "roofline_model.proto",
        "source_info.proto",
        "steps_db.proto",
        "task.proto",
        "tf_data_stats.proto",
        "tf_function.proto",
        "tf_stats.proto",
        "tfstreamz.proto",
        "topology.proto",
        "tpu_input_pipeline.proto",
        "trace_events.proto",
        "trace_events_old.proto",
        "trace_events_raw.proto",
    ],
    visibility = [
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:__pkg__",
    ],
)
