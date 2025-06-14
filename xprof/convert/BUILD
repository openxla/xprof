load("//defs:backend_defs.bzl", "if_oss", "tf_profiler_alias")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],  # TODO(matthurd): Update to profiler:internal after xprof migration.
    licenses = ["notice"],
)

cc_library(
    name = "xplane_to_op_metrics_db",
    srcs = ["xplane_to_op_metrics_db.cc"],
    hdrs = ["xplane_to_op_metrics_db.h"],
    deps = [
        ":op_metrics_db_combiner",
        ":op_stack",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//xprof/utils:cost_utils",
        "@org_xprof//xprof/utils:gpu_event_stats",
        "@org_xprof//xprof/utils:hlo_module_map",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@org_xprof//xprof/utils:op_utils",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:trace_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "xplane_to_op_metrics_db_test",
    size = "small",
    srcs = ["xplane_to_op_metrics_db_test.cc"],
    deps = [
        ":xplane_to_op_metrics_db",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//xprof/utils:hlo_cost_analysis_wrapper",
        "@org_xprof//xprof/utils:hlo_module_map",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@org_xprof//xprof/utils:xprof_gpu_cost_analysis",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_test_utils",
    ],
)

cc_library(
    name = "op_metrics_db_combiner",
    srcs = ["op_metrics_db_combiner.cc"],
    hdrs = ["op_metrics_db_combiner.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:source_info_proto_cc",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "op_metrics_to_record",
    srcs = ["op_metrics_to_record.cc"],
    hdrs = ["op_metrics_to_record.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "op_stack",
    hdrs = ["op_stack.h"],
    deps = [
        "@xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "op_stats_to_hlo_stats",
    srcs = ["op_stats_to_hlo_stats.cc"],
    hdrs = ["op_stats_to_hlo_stats.h"],
    deps = [
        ":data_table_utils",
        ":op_metrics_to_record",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:hlo_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:source_info_proto_cc",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
    ],
)

cc_library(
    name = "op_stats_to_roofline_model",
    srcs = ["op_stats_to_roofline_model.cc"],
    hdrs = ["op_stats_to_roofline_model.h"],
    deps = [
        ":op_metrics_db_combiner",
        ":op_metrics_to_record",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:roofline_model_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:source_info_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//xprof/convert:data_table_utils",
        "@org_xprof//xprof/utils:diagnostics",
        "@org_xprof//xprof/utils:roofline_model_utils",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "op_stats_to_op_profile",
    srcs = ["op_stats_to_op_profile.cc"],
    hdrs = ["op_stats_to_op_profile.h"],
    deps = [
        ":op_profile_builder",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_profile_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "op_stats_to_overview_page",
    srcs = ["op_stats_to_overview_page.cc"],
    hdrs = ["op_stats_to_overview_page.h"],
    deps = [
        ":op_metrics_to_record",
        ":op_stats_to_input_pipeline_analysis",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:input_pipeline_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:kernel_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:power_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:tf_function_proto_cc",
        "@org_xprof//xprof/utils:diagnostics",
        "@org_xprof//xprof/utils:hardware_type_utils",
        "@org_xprof//xprof/utils:html_utils",
        "@org_xprof//xprof/utils:kernel_stats_utils",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:format_utils",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
    ],
)

cc_library(
    name = "op_stats_to_pod_stats",
    srcs = ["op_stats_to_pod_stats.cc"],
    hdrs = ["op_stats_to_pod_stats.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:pod_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//xprof/utils:diagnostics",
        "@org_xprof//xprof/utils:event_span",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_test(
    name = "op_stats_to_pod_stats_test",
    srcs = ["op_stats_to_pod_stats_test.cc"],
    deps = [
        ":op_stats_to_pod_stats",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:diagnostics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//xprof/utils:diagnostics",
        "@org_xprof//xprof/utils:event_span",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "op_stats_to_pod_viewer",
    srcs = ["op_stats_to_pod_viewer.cc"],
    hdrs = ["op_stats_to_pod_viewer.h"],
    deps = [
        ":op_stats_to_pod_stats",
        "@com_google_absl//absl/log:check",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:pod_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:pod_viewer_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//xprof/utils:diagnostics",
    ],
)

cc_test(
    name = "op_stats_to_pod_viewer_test",
    srcs = ["op_stats_to_pod_viewer_test.cc"],
    deps = [
        ":op_stats_to_pod_viewer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:diagnostics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:pod_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//xprof/utils:diagnostics",
        "@org_xprof//xprof/utils:event_span",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_test(
    name = "xplane_to_trace_container_test",
    srcs = ["xplane_to_trace_container_test.cc"],
    deps = [
        ":xplane_to_trace_container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//xprof/utils:tensorflow_utils",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "op_stats_to_input_pipeline_analysis",
    srcs = ["op_stats_to_input_pipeline_analysis.cc"],
    hdrs = ["op_stats_to_input_pipeline_analysis.h"],
    deps = [
        ":op_metrics_to_record",
        ":profile_time_breakdown",
        ":step_events_to_steps_db",
        ":tpu_input_pipeline_analysis_constants",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:input_pipeline_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:tpu_input_pipeline_proto_cc",
        "@org_xprof//xprof/utils:diagnostics",
        "@org_xprof//xprof/utils:event_span",
        "@org_xprof//xprof/utils:html_utils",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@org_xprof//xprof/utils:tpu_step_breakdown_utils",
        "@org_xprof//xprof/utils:tpu_step_details_utils",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:format_utils",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/util:stats_calculator_portable",
    ],
)

cc_test(
    name = "op_stats_to_input_pipeline_analysis_test",
    srcs = ["op_stats_to_input_pipeline_analysis_test.cc"],
    deps = [
        ":op_stats_to_input_pipeline_analysis",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//xprof/utils:event_span",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "op_stats_to_tf_stats",
    srcs = ["op_stats_to_tf_stats.cc"],
    hdrs = ["op_stats_to_tf_stats.h"],
    deps = [
        ":data_table_utils",
        ":model_tracker",
        ":op_metrics_to_record",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:tf_stats_proto_cc",
        "@org_xprof//xprof/utils:kernel_stats_utils",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_test(
    name = "op_stats_to_tf_stats_test",
    size = "small",
    srcs = ["op_stats_to_tf_stats_test.cc"],
    deps = [
        ":op_stats_to_tf_stats",
        ":xplane_to_op_stats",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:tf_stats_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_test_utils",
    ],
)

cc_library(
    name = "step_events_to_steps_db",
    srcs = ["step_events_to_steps_db.cc"],
    hdrs = ["step_events_to_steps_db.h"],
    deps = [
        ":op_metrics_db_combiner",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//xprof/utils:event_span",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:macros",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "executor_interface",
    hdrs = ["executor.h"],
    deps = [
    ],
)

cc_library(
    name = "xprof_thread_pool_executor",
    srcs = ["xprof_thread_pool_executor.cc"],
    hdrs = ["xprof_thread_pool_executor.h"],
    deps = [
        ":executor_interface",
        "@com_google_absl//absl/log",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_library(
    name = "xplane_to_op_stats",
    srcs = ["xplane_to_op_stats.cc"],
    hdrs = ["xplane_to_op_stats.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":duty_cycle_combiner",
        ":duty_cycle_tracker",
        ":model_tracker",
        ":op_metrics_db_combiner",
        ":step_events_to_steps_db",
        ":xplane_to_kernel_stats_db",
        ":xplane_to_op_metrics_db",
        ":xplane_to_step_events",
        ":xplane_to_tf_functions",
        ":xprof_thread_pool_executor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:diagnostics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:tf_function_proto_cc",
        "@org_xprof//xprof/utils:device_caps_utils",
        "@org_xprof//xprof/utils:event_span",
        "@org_xprof//xprof/utils:gpu_event_stats",
        "@org_xprof//xprof/utils:hardware_type_utils",
        "@org_xprof//xprof/utils:hlo_cost_analysis_wrapper",
        "@org_xprof//xprof/utils:hlo_module_map",
        "@org_xprof//xprof/utils:hlo_proto_map",
        "@org_xprof//xprof/utils:kernel_stats_utils",
        "@org_xprof//xprof/utils:op_utils",
        "@org_xprof//xprof/utils:xprof_gpu_cost_analysis_types",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:tpu_xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@org_xprof//xprof/utils:xprof_gpu_cost_analysis_registration",
    ],
)

cc_library(
    name = "multi_xplanes_to_op_stats",
    srcs = ["multi_xplanes_to_op_stats.cc"],
    hdrs = ["multi_xplanes_to_op_stats.h"],
    deps = [
        ":op_stats_combiner",
        ":preprocess_single_host_xplane",
        ":repository",
        ":xplane_to_op_stats",
        "@com_google_absl//absl/status",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//xprof/utils:hardware_type_utils",
        "@org_xprof//xprof/utils:step_intersection",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:types",
    ],
)

cc_test(
    name = "xplane_to_op_stats_test",
    size = "small",
    srcs = ["xplane_to_op_stats_test.cc"],
    tags = [
        "asan",
        "tsan",
    ],
    deps = [
        ":duty_cycle_tracker",
        ":multi_xplanes_to_op_stats",
        ":repository",
        ":step_events_to_steps_db",
        ":xplane_to_op_stats",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:diagnostics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:tf_function_proto_cc",
        "@org_xprof//xprof/utils:hlo_proto_map",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_test_utils",
    ],
)

cc_library(
    name = "xplane_to_step_events",
    srcs = ["xplane_to_step_events.cc"],
    hdrs = ["xplane_to_step_events.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//xprof/utils:event_span",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:tpu_xplane_utils",
        "@xla//xla/tsl/profiler/utils:trace_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "xplane_to_step_events_test",
    size = "small",
    srcs = ["xplane_to_step_events_test.cc"],
    deps = [
        ":xplane_to_step_events",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//xprof/utils:derived_timeline",
        "@org_xprof//xprof/utils:event_span",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_test_utils",
    ],
)

cc_library(
    name = "xplane_to_kernel_stats_db",
    srcs = ["xplane_to_kernel_stats_db.cc"],
    hdrs = ["xplane_to_kernel_stats_db.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:kernel_stats_proto_cc",
        "@org_xprof//xprof/convert:data_table_utils",
        "@org_xprof//xprof/utils:gpu_event_stats",
        "@org_xprof//xprof/utils:kernel_stats_utils",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:trace_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "xplane_to_kernel_stats_db_test",
    size = "small",
    srcs = ["xplane_to_kernel_stats_db_test.cc"],
    deps = [
        ":xplane_to_kernel_stats_db",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:kernel_stats_proto_cc",
        "@org_xprof//xprof/utils:kernel_stats_utils",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_test_utils",
    ],
)

cc_library(
    name = "xplane_to_tf_functions",
    srcs = ["xplane_to_tf_functions.cc"],
    hdrs = ["xplane_to_tf_functions.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:tf_function_proto_cc",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "xplane_to_tf_functions_test",
    size = "small",
    srcs = ["xplane_to_tf_functions_test.cc"],
    deps = [
        ":xplane_to_tf_functions",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:tf_function_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_test_utils",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "xplane_to_memory_profile",
    srcs = ["xplane_to_memory_profile.cc"],
    hdrs = ["xplane_to_memory_profile.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:memory_profile_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:tensorflow_datatypes_proto_cc",
        "@org_xprof//xprof/utils:tensorflow_utils",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "xplane_to_memory_profile_test",
    size = "small",
    srcs = ["xplane_to_memory_profile_test.cc"],
    deps = [
        ":xplane_to_memory_profile",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:memory_profile_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:xplane_test_utils",
    ],
)

cc_library(
    name = "op_stats_combiner",
    srcs = ["op_stats_combiner.cc"],
    hdrs = ["op_stats_combiner.h"],
    deps = [
        ":op_metrics_db_combiner",
        ":xplane_to_tf_functions",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@org_xprof//plugin/xprof/protobuf:diagnostics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:kernel_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:power_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:topology_proto_cc",
        "@org_xprof//xprof/utils:hardware_type_utils",
        "@org_xprof//xprof/utils:kernel_stats_utils",
        "@org_xprof//xprof/utils:step_intersection",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:types",
    ],
)

cc_test(
    name = "op_stats_combiner_test",
    srcs = ["op_stats_combiner_test.cc"],
    deps = [
        ":op_stats_combiner",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@org_xprof//xprof/utils:step_intersection",
        "@xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "preprocess_single_host_xplane",
    srcs = ["preprocess_single_host_xplane.cc"],
    hdrs = ["preprocess_single_host_xplane.h"],
    # copybara:uncomment visibility = ["@org_tensorflow//tensorflow/core/profiler:internal"],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_xprof//xprof/utils:derived_timeline",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:preprocess_xplane",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "xplane_to_tools_data",
    srcs = ["xplane_to_tools_data.cc"],
    hdrs = ["xplane_to_tools_data.h"],
    deps = [
        ":compute_inference_latency",
        ":hlo_to_tools_data",
        ":multi_xplanes_to_op_stats",
        ":multi_xspace_to_inference_stats",
        ":op_stats_to_hlo_stats",
        ":op_stats_to_input_pipeline_analysis",
        ":op_stats_to_op_profile",
        ":op_stats_to_overview_page",
        ":op_stats_to_pod_viewer",
        ":op_stats_to_roofline_model",
        ":op_stats_to_tf_stats",
        ":preprocess_single_host_xplane",
        ":process_megascale_dcn",
        ":repository",
        ":tool_options",
        ":xplane_to_dcn_collective_stats",
        ":xplane_to_hlo",
        ":xplane_to_kernel_stats_db",
        ":xplane_to_memory_profile",
        ":xplane_to_op_stats",
        ":xplane_to_tf_data_stats",
        ":xplane_to_tool_names",
        ":xplane_to_trace_container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:dcn_slack_analysis_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:inference_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:input_pipeline_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:kernel_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_profile_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:roofline_model_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:tf_data_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:tf_stats_proto_cc",
        "@org_xprof//xprof/convert/trace_viewer:trace_events_to_json",
        "@org_xprof//xprof/convert/trace_viewer:trace_viewer_visibility",
        "@org_xprof//xprof/utils:hardware_type_utils",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/profiler/convert:xplane_to_trace_events",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "xplane_to_tf_data_stats",
    srcs = ["xplane_to_tf_data_stats.cc"],
    hdrs = ["xplane_to_tf_data_stats.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_xprof//plugin/xprof/protobuf:tf_data_stats_proto_cc",
        "@org_xprof//xprof/utils:html_utils",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/platform:macros",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "xplane_to_tf_data_stats_test",
    size = "small",
    srcs = ["xplane_to_tf_data_stats_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    deps = [
        ":xplane_to_tf_data_stats",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:tf_data_stats_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_test_utils",
    ],
)

cc_library(
    name = "hlo_to_tools_data",
    srcs = ["hlo_to_tools_data.cc"],
    hdrs = ["hlo_to_tools_data.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":hlo_proto_to_graph_view",
        ":hlo_proto_to_memory_visualization_utils",
        ":repository",
        ":tool_options",
        ":xplane_to_hlo",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:memory_viewer_preprocess_proto_cc",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_proto_to_memory_visualization_utils",
    srcs = ["hlo_proto_to_memory_visualization_utils.cc"],
    hdrs = ["hlo_proto_to_memory_visualization_utils.h"],
    # copybara:uncomment visibility = ["@org_tensorflow//tensorflow/core/profiler/protobuf:memory_viewer_friends"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@org_xprof//plugin/xprof/protobuf:memory_viewer_preprocess_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:source_info_proto_cc",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:types",
    ],
)

cc_test(
    name = "hlo_proto_to_memory_visualization_utils_test",
    srcs = ["hlo_proto_to_memory_visualization_utils_test.cc"],
    deps = [
        ":hlo_proto_to_memory_visualization_utils",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:memory_viewer_preprocess_proto_cc",
        "@org_xprof//xprof/utils:tensorflow_utils",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "xplane_to_hlo",
    srcs = ["xplane_to_hlo.cc"],
    hdrs = ["xplane_to_hlo.h"],
    deps = [
        ":repository",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_xprof//xprof/utils:hlo_proto_map",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/profiler/utils:file_system_utils",
    ],
)

cc_library(
    name = "op_profile_builder",
    srcs = ["op_profile_builder.cc"],
    hdrs = ["op_profile_builder.h"],
    deps = [
        ":op_metrics_db_combiner",
        ":op_metrics_to_record",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_profile_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:source_info_proto_cc",
        "@org_xprof//xprof/utils:op_metrics_db_utils",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "hlo_proto_to_graph_view",
    srcs = ["hlo_proto_to_graph_view.cc"],
    hdrs = ["hlo_proto_to_graph_view.h"],
    deps = [
        ":tool_options",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_github_nlohmann_json//:json",
        "@com_google_protobuf//:protobuf",
        # copybara:uncomment "//third_party/protobuf/util:json_util",
        # copybara:uncomment(b/360874576) "@org_tensorflow//tensorflow/compiler/mlir/lite/experimental/google/tooling/hlo_adapter:direct_hlo_to_json_graph_convert",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_graph_dumper",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@org_xprof//xprof/utils:hlo_module_utils",
        "@org_xprof//xprof/utils:hlo_proto_to_module",
    ],
)

cc_test(
    name = "hlo_proto_to_graph_view_test",
    size = "small",
    srcs = ["hlo_proto_to_graph_view_test.cc"],
    deps = [
        ":hlo_proto_to_graph_view",
        ":tool_options",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/service:hlo_graph_dumper",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:status_matchers",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "tool_options",
    hdrs = ["tool_options.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "repository",
    srcs = ["repository.cc"],
    hdrs = ["repository.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_xprof//xprof/utils:hlo_module_map",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/profiler/utils:file_system_utils",
    ],
)

cc_test(
    name = "repository_test",
    size = "small",
    srcs = ["repository_test.cc"],
    deps = [
        ":repository",
        "//testing/lib/sponge:undeclared_outputs",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "xplane_to_tool_names",
    srcs = ["xplane_to_tool_names.cc"],
    hdrs = ["xplane_to_tool_names.h"],
    deps = [
        ":repository",
        ":xplane_to_dcn_collective_stats",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "xplane_to_tool_names_test",
    size = "small",
    srcs = ["xplane_to_tool_names_test.cc"],
    deps = [
        ":repository",
        ":xplane_to_tool_names",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/backends/profiler/cpu:metadata_utils",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "xplane_to_trace_container",
    srcs = ["xplane_to_trace_container.cc"],
    hdrs = ["xplane_to_trace_container.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:trace_events_raw_proto_cc",
        "@org_xprof//xprof/convert/trace_viewer:trace_event_arguments_builder",
        "@org_xprof//xprof/convert/trace_viewer:trace_events",
        "@org_xprof//xprof/convert/trace_viewer:trace_events_util",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:trace_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "dcn_utils",
    srcs = ["dcn_utils.cc"],
    hdrs = ["dcn_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "dcn_utils_test",
    srcs = ["dcn_utils_test.cc"],
    deps = [
        ":dcn_utils",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "dcn_analysis",
    srcs = ["dcn_analysis.cc"],
    hdrs = ["dcn_analysis.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":dcn_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:tpu_xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "process_megascale_dcn",
    srcs = ["process_megascale_dcn.cc"],
    hdrs = ["process_megascale_dcn.h"],
    deps = [
        ":dcn_analysis",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:tpu_xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "dcn_analysis_test",
    srcs = ["dcn_analysis_test.cc"],
    deps = [
        ":dcn_analysis",
        ":dcn_utils",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "xspace_to_dcn_slack_analysis",
    srcs = ["xspace_to_dcn_slack_analysis.cc"],
    hdrs = ["xspace_to_dcn_slack_analysis.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:dcn_collective_info_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:dcn_slack_analysis_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:topology_proto_cc",
        "@org_xprof//xprof/utils:hlo_module_utils",
        "@org_xprof//xprof/utils:hlo_proto_map",
        "@org_xprof//xprof/utils:hlo_proto_to_module",
        "@tsl//tsl/platform:regexp",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla:shape_util",
        "@xla//xla:side_effect_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:tpu_xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "dcn_slack_analysis_combiner",
    srcs = ["dcn_slack_analysis_combiner.cc"],
    hdrs = ["dcn_slack_analysis_combiner.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@org_xprof//plugin/xprof/protobuf:dcn_slack_analysis_proto_cc",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "xplane_to_dcn_collective_stats",
    srcs = ["xplane_to_dcn_collective_stats.cc"],
    hdrs = ["xplane_to_dcn_collective_stats.h"],
    deps = [
        ":dcn_slack_analysis_combiner",
        ":repository",
        ":xspace_to_dcn_slack_analysis",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:dcn_slack_analysis_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "xplane_to_dcn_collective_stats_test",
    size = "small",
    srcs = ["xplane_to_dcn_collective_stats_test.cc"],
    deps = [
        ":repository",
        ":xplane_to_dcn_collective_stats",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:dcn_slack_analysis_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "inference_stats",
    srcs = ["inference_stats.cc"],
    hdrs = ["inference_stats.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@org_xprof//plugin/xprof/protobuf:inference_stats_proto_cc",
        "@org_xprof//xprof/utils:event_span",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/profiler/utils:device_utils",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "inference_stats_sampler",
    srcs = ["inference_stats_sampler.cc"],
    hdrs = ["inference_stats_sampler.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@org_xprof//plugin/xprof/protobuf:inference_stats_proto_cc",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "inference_stats_grouping",
    srcs = ["inference_stats_grouping.cc"],
    hdrs = ["inference_stats_grouping.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@org_xprof//plugin/xprof/protobuf:inference_stats_proto_cc",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "inference_stats_combiner",
    srcs = ["inference_stats_combiner.cc"],
    hdrs = ["inference_stats_combiner.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@org_xprof//plugin/xprof/protobuf:inference_stats_proto_cc",
        "@xla//xla/tsl/lib/gtl:map_util",
    ],
)

cc_library(
    name = "multi_xspace_to_inference_stats",
    srcs = ["multi_xspace_to_inference_stats.cc"],
    hdrs = ["multi_xspace_to_inference_stats.h"],
    deps = [
        ":inference_stats",
        ":inference_stats_combiner",
        ":inference_stats_grouping",
        ":inference_stats_sampler",
        ":preprocess_single_host_xplane",
        ":repository",
        ":xplane_to_step_events",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@org_xprof//plugin/xprof/protobuf:inference_stats_proto_cc",
        "@org_xprof//xprof/utils:event_span",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/profiler/utils:device_utils",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:tpu_xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
    ],
)

cc_test(
    name = "inference_stats_grouping_test",
    srcs = ["inference_stats_grouping_test.cc"],
    tags = [
        "no_oss",
    ],
    deps = [
        ":inference_stats_grouping",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:inference_stats_proto_cc",
        "@xla//xla/tests:test_utils",
    ],
)

cc_test(
    name = "inference_stats_sampler_test",
    srcs = ["inference_stats_sampler_test.cc"],
    tags = [
        "no_oss",
    ],
    deps = [
        ":inference_stats_sampler",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:inference_stats_proto_cc",
        "@xla//xla/tests:test_utils",
    ],
)

cc_library(
    name = "compute_inference_latency",
    srcs = ["compute_inference_latency.cc"],
    hdrs = ["compute_inference_latency.h"],
    visibility = ["//perftools/accelerators/xprof/convert:__pkg__"],
    deps = [
        "@org_xprof//plugin/xprof/protobuf:inference_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "profile_time_breakdown",
    srcs = ["profile_time_breakdown.cc"],
    hdrs = ["profile_time_breakdown.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "tpu_input_pipeline_analysis_constants",
    srcs = [tf_profiler_alias("@org_xprof//xprof/convert/", "tpu_input_pipeline_analysis_constants.cc")],
    hdrs = ["tpu_input_pipeline_analysis_constants.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@xla//xla/tsl/platform:macros",
    ],
)

cc_library(
    name = "duty_cycle_tracker",
    srcs = ["duty_cycle_tracker.cc"],
    hdrs = ["duty_cycle_tracker.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:check",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "duty_cycle_combiner",
    hdrs = ["duty_cycle_combiner.h"],
    deps = [
        ":duty_cycle_tracker",
        "@com_google_absl//absl/container:flat_hash_map",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_test(
    name = "duty_cycle_tracker_test",
    srcs = ["duty_cycle_tracker_test.cc"],
    deps = [
        ":duty_cycle_tracker",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:test_benchmark",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_test(
    name = "compute_inference_latency_test",
    srcs = ["compute_inference_latency_test.cc"],
    deps = [
        ":compute_inference_latency",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:inference_stats_proto_cc",
    ],
)

cc_test(
    name = "duty_cycle_combiner_test",
    srcs = ["duty_cycle_combiner_test.cc"],
    deps = [
        ":duty_cycle_combiner",
        ":duty_cycle_tracker",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "data_table_utils",
    hdrs = ["data_table_utils.h"],
    deps = [
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "model_tracker",
    srcs = ["model_tracker.cc"],
    hdrs = ["model_tracker.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@org_xprof//xprof/utils:hlo_module_map",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
    ],
)

cc_test(
    name = "data_table_utils_test",
    srcs = ["data_table_utils_test.cc"],
    deps = [
        ":data_table_utils",
        "@com_github_nlohmann_json//:json",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
