package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "@org_tensorflow//tensorflow/core/profiler:friends",
    ],
)

cc_library(
    name = "diagnostics",
    srcs = ["diagnostics.cc"],
    hdrs = ["diagnostics.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:macros",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:diagnostics_proto_cc",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:steps_db_proto_cc",
    ],
)

cc_library(
    name = "event_span",
    srcs = ["event_span.cc"],
    hdrs = ["event_span.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/lib/gtl:map_util",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/profiler/utils:timespan",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:steps_db_proto_cc",
    ],
)

cc_library(
    name = "hardware_type_utils",
    srcs = ["hardware_type_utils.cc"],
    hdrs = ["hardware_type_utils.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/profiler/utils:math_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:hardware_types_proto_cc",
    ],
)

cc_test(
    name = "hardware_type_utils_test",
    srcs = ["hardware_type_utils_test.cc"],
    deps = [
        ":hardware_type_utils",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "html_utils",
    hdrs = ["html_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_metrics_db_utils",
    srcs = ["op_metrics_db_utils.cc"],
    hdrs = ["op_metrics_db_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:macros",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/profiler/utils:math_utils",
        "@local_xla//xla/tsl/profiler/utils:tf_op_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:source_info_proto_cc",
    ],
)

cc_test(
    name = "op_metrics_db_utils_test",
    srcs = ["op_metrics_db_utils_test.cc"],
    deps = [
        ":op_metrics_db_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@local_xla//xla/tsl/profiler/utils:xplane_builder",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_metrics_proto_cc",
    ],
)

cc_library(
    name = "op_utils",
    srcs = ["op_utils.cc"],
    hdrs = ["op_utils.h"],
    deps = [
        ":hlo_module_map",
        ":op_metrics_db_utils",
        ":performance_info_wrapper",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/profiler/convert:xla_op_utils",
        "@local_xla//xla/tsl/profiler/utils:tf_op_utils",
        "@local_xla//xla/tsl/profiler/utils:timespan",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:source_info_proto_cc",
        "@org_xprof//xprof/convert:op_metrics_db_combiner",
    ],
)

cc_library(
    name = "cost_utils",
    srcs = ["cost_utils.cc"],
    hdrs = ["cost_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/profiler/utils:tf_op_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/grappler/costs:cost_estimator",
        "@org_tensorflow//tensorflow/core/grappler/costs:op_context",
        "@org_tensorflow//tensorflow/core/grappler/costs:op_level_cost_estimator",
        "@org_tensorflow//tensorflow/core/grappler/costs:op_performance_data_cc",
    ],
)

cc_library(
    name = "host_offload_utils",
    srcs = ["host_offload_utils.cc"],
    hdrs = ["host_offload_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/tsl/profiler/utils:timespan",
        "@local_xla//xla/tsl/profiler/utils:trace_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_builder",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "derived_timeline",
    srcs = ["derived_timeline.cc"],
    hdrs = ["derived_timeline.h"],
    deps = [
        ":gpu_event_stats",
        ":hlo_module_map",
        ":hlo_proto_map",
        ":host_offload_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/lib/gtl:map_util",
        "@local_xla//xla/tsl/profiler/convert:xla_op_utils",
        "@local_xla//xla/tsl/profiler/utils:device_utils",
        "@local_xla//xla/tsl/profiler/utils:group_events",
        "@local_xla//xla/tsl/profiler/utils:math_utils",
        "@local_xla//xla/tsl/profiler/utils:tf_op_utils",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@local_xla//xla/tsl/profiler/utils:timespan",
        "@local_xla//xla/tsl/profiler/utils:tpu_xplane_utils",
        "@local_xla//xla/tsl/profiler/utils:trace_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_builder",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
    ],
)

cc_test(
    name = "derived_timeline_test",
    srcs = ["derived_timeline_test.cc"],
    deps = [
        ":derived_timeline",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/profiler/utils:group_events",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@local_xla//xla/tsl/profiler/utils:trace_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_builder",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_test_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "kernel_stats_utils",
    srcs = ["kernel_stats_utils.cc"],
    hdrs = ["kernel_stats_utils.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:types",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:kernel_stats_proto_cc",
    ],
)

cc_test(
    name = "kernel_stats_utils_test",
    srcs = ["kernel_stats_utils_test.cc"],
    deps = [
        ":kernel_stats_utils",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/backends/profiler/gpu:cupti_buffer_events",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:kernel_stats_proto_cc",
    ],
)

cc_library(
    name = "tfstreamz_utils",
    srcs = ["tfstreamz_utils.cc"],
    hdrs = ["tfstreamz_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/lib/monitoring:collected_metrics",
        "@local_xla//xla/tsl/lib/monitoring:metric_def",
        "@local_xla//xla/tsl/lib/monitoring:types",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/profiler/utils:xplane_builder",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tfstreamz_proto_cc",
    ],
)

cc_library(
    name = "step_intersection",
    srcs = ["step_intersection.cc"],
    hdrs = ["step_intersection.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/lib/gtl:map_util",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/profiler/utils:timespan",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:steps_db_proto_cc",
    ],
)

cc_test(
    name = "step_intersection_test",
    srcs = ["step_intersection_test.cc"],
    deps = [
        ":step_intersection",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "device_caps_utils",
    srcs = ["device_caps_utils.cc"],
    hdrs = ["device_caps_utils.h"],
    deps = [
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@local_xla//xla/tsl/profiler/utils:xplane_builder",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:hardware_types_proto_cc",
    ],
)

cc_library(
    name = "gpu_event_stats",
    srcs = ["gpu_event_stats.cc"],
    hdrs = ["gpu_event_stats.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "hlo_proto_map",
    srcs = ["hlo_proto_map.cc"],
    hdrs = ["hlo_proto_map.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/profiler/convert:xla_op_utils",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "hlo_proto_to_module",
    srcs = ["hlo_proto_to_module.cc"],
    hdrs = ["hlo_proto_to_module.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@local_xla//xla:util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_module_map",
    srcs = ["hlo_module_map.cc"],
    hdrs = ["hlo_module_map.h"],
    deps = [
        ":hlo_cost_analysis_wrapper",
        ":hlo_module_utils",
        ":hlo_proto_map",
        ":hlo_proto_to_module",
        ":performance_info_wrapper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_cost_analysis",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/profiler/convert:xla_op_utils",
    ],
)

cc_library(
    name = "hlo_module_utils",
    hdrs = ["hlo_module_utils.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/tsl/profiler/convert:xla_op_utils",
    ],
)

cc_test(
    name = "hlo_module_utils_test",
    srcs = ["hlo_module_utils_test.cc"],
    deps = [
        ":hlo_module_utils",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/tests:hlo_test_base",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "xprof_gpu_cost_analysis",
    srcs = ["xprof_gpu_cost_analysis.cc"],
    hdrs = ["xprof_gpu_cost_analysis.h"],
    deps = [
        ":cost_utils",
        ":hlo_cost_analysis_wrapper",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_cost_analysis",
        "@local_xla//xla/service/gpu:cublas_cudnn",
        "@local_xla//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "@local_xla//xla/tsl/platform:errors",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_metrics_proto_cc",
    ],
)

cc_library(
    name = "tpu_step_breakdown_utils",
    hdrs = ["tpu_step_breakdown_utils.h"],
    deps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:steps_db_proto_cc"],
)

cc_library(
    name = "tpu_step_details_utils",
    hdrs = ["tpu_step_details_utils.h"],
    deps = ["@org_xprof//plugin/tensorboard_plugin_profile/protobuf:tpu_input_pipeline_proto_cc"],
)

cc_test(
    name = "xprof_gpu_cost_analysis_test",
    srcs = ["xprof_gpu_cost_analysis_test.cc"],
    deps = [
        ":xprof_gpu_cost_analysis",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/testlib:test_helpers",
        "@local_xla//xla/service:hlo_cost_analysis",
        "@local_xla//xla/tests:hlo_test_base",
        "@local_xla//xla/tests:xla_internal_test_main",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_cost_analysis_wrapper",
    srcs = ["hlo_cost_analysis_wrapper.cc"],
    hdrs = ["hlo_cost_analysis_wrapper.h"],
    deps = [
        ":cost_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_cost_analysis",
        "@local_xla//xla/tsl/profiler/convert:xla_op_utils",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_metrics_proto_cc",
    ],
)

cc_test(
    name = "hlo_cost_analysis_wrapper_test",
    srcs = ["hlo_cost_analysis_wrapper_test.cc"],
    deps = [
        ":hlo_cost_analysis_wrapper",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/parser:hlo_parser",
        "@local_xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "performance_info_wrapper",
    srcs = ["performance_info_wrapper.cc"],
    hdrs = ["performance_info_wrapper.h"],
    deps = [
        ":hlo_cost_analysis_wrapper",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:op_metrics_proto_cc",
    ],
)

cc_test(
    name = "performance_info_wrapper_test",
    srcs = ["performance_info_wrapper_test.cc"],
    deps = [
        ":hlo_cost_analysis_wrapper",
        ":hlo_module_map",
        ":performance_info_wrapper",
        ":xprof_gpu_cost_analysis",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/parser:hlo_parser",
        "@local_xla//xla/tsl/platform:test",
    ],
)
