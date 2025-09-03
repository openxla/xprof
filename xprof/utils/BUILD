# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")

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
    name = "xprof_gpu_cost_analysis_registration",
    srcs = ["xprof_gpu_cost_analysis_registration.cc"],
    deps = [
        ":function_registry",
        ":hlo_cost_analysis_wrapper",
        ":xprof_gpu_cost_analysis",
        ":xprof_gpu_cost_analysis_types",
    ],
    alwayslink = 1,
)

cc_library(
    name = "diagnostics",
    srcs = ["diagnostics.cc"],
    hdrs = ["diagnostics.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:diagnostics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:op_stats_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@xla//xla/tsl/platform:macros",
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
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:timespan",
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
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
    ],
)

cc_test(
    name = "hardware_type_utils_test",
    srcs = ["hardware_type_utils_test.cc"],
    deps = [
        ":hardware_type_utils",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/profiler/utils:math_utils",
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
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:source_info_proto_cc",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:macros",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_test(
    name = "op_metrics_db_utils_test",
    srcs = ["op_metrics_db_utils_test.cc"],
    deps = [
        ":op_metrics_db_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
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
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@org_xprof//plugin/xprof/protobuf:source_info_proto_cc",
        "@org_xprof//xprof/convert:op_metrics_db_combiner",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "cost_utils",
    hdrs = ["cost_utils.h"],
    deps = [
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
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
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla:shape_util",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:trace_utils",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
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
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@org_xprof//xprof/convert:xprof_thread_pool_executor",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:device_utils",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:math_utils",
        "@xla//xla/tsl/profiler/utils:tf_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:timespan",
        "@xla//xla/tsl/profiler/utils:tpu_xplane_utils",
        "@xla//xla/tsl/profiler/utils:trace_utils",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
        "@xla//xla/tsl/util:stats_calculator_portable",
    ],
)

cc_test(
    name = "derived_timeline_test",
    srcs = ["derived_timeline_test.cc"],
    tags = [
        "asan",
        "tsan",
    ],
    deps = [
        ":derived_timeline",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:group_events",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:trace_utils",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_test_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
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
        "@org_xprof//plugin/xprof/protobuf:kernel_stats_proto_cc",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:types",
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
        "@org_xprof//plugin/xprof/protobuf:kernel_stats_proto_cc",
        "@xla//xla/backends/profiler/gpu:cupti_buffer_events",
    ],
)

cc_library(
    name = "tfstreamz_utils",
    srcs = ["tfstreamz_utils.cc"],
    hdrs = ["tfstreamz_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:tfstreamz_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/lib/monitoring:collected_metrics",
        "@xla//xla/tsl/lib/monitoring:metric_def",
        "@xla//xla/tsl/lib/monitoring:types",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
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
        "@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc",
        "@xla//xla/tsl/lib/gtl:map_util",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_test(
    name = "step_intersection_test",
    srcs = ["step_intersection_test.cc"],
    deps = [
        ":step_intersection",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "device_caps_utils",
    srcs = ["device_caps_utils.cc"],
    hdrs = ["device_caps_utils.h"],
    deps = [
        "@org_xprof//plugin/xprof/protobuf:hardware_types_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "gpu_event_stats",
    srcs = ["gpu_event_stats.cc"],
    hdrs = ["gpu_event_stats.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
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
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
        "@xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
        "@xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "hlo_proto_to_module",
    srcs = ["hlo_proto_to_module.cc"],
    hdrs = ["hlo_proto_to_module.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla:util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/platform:statusor",
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
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:traceme_encode",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_cost_analysis",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
    ],
)

cc_library(
    name = "hlo_module_utils",
    hdrs = ["hlo_module_utils.h"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/tsl/profiler/convert:xla_op_utils",
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
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/tests:hlo_test_base",
        "@xla//xla/tsl/platform:statusor",
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
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_cost_analysis",
        "@xla//xla/service/gpu:cublas_cudnn",
        "@xla//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "tpu_step_breakdown_utils",
    hdrs = ["tpu_step_breakdown_utils.h"],
    deps = ["@org_xprof//plugin/xprof/protobuf:steps_db_proto_cc"],
)

cc_library(
    name = "tpu_step_details_utils",
    hdrs = ["tpu_step_details_utils.h"],
    deps = ["@org_xprof//plugin/xprof/protobuf:tpu_input_pipeline_proto_cc"],
)

cc_test(
    name = "xprof_gpu_cost_analysis_test",
    srcs = ["xprof_gpu_cost_analysis_test.cc"],
    deps = [
        ":xprof_gpu_cost_analysis",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/testlib:test_helpers",
        "@xla//xla/service:hlo_cost_analysis",
        "@xla//xla/tests:hlo_test_base",
        "@xla//xla/tests:xla_internal_test_main",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_cost_analysis_wrapper",
    srcs = ["hlo_cost_analysis_wrapper.cc"],
    hdrs = ["hlo_cost_analysis_wrapper.h"],
    deps = [
        ":cost_utils",
        ":function_registry",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/service:hlo_cost_analysis",
    ],
)

cc_test(
    name = "hlo_cost_analysis_wrapper_test",
    srcs = ["hlo_cost_analysis_wrapper_test.cc"],
    deps = [
        ":hlo_cost_analysis_wrapper",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "performance_info_wrapper",
    srcs = ["performance_info_wrapper.cc"],
    hdrs = ["performance_info_wrapper.h"],
    deps = [
        ":hlo_cost_analysis_wrapper",
        "@com_google_absl//absl/memory",
        "@org_xprof//plugin/xprof/protobuf:op_metrics_proto_cc",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
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
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "tensorflow_utils",
    srcs = ["tensorflow_utils.cc"],
    hdrs = ["tensorflow_utils.h"],
    visibility = ["//xprof:__subpackages__"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@org_xprof//plugin/xprof/protobuf:tensorflow_datatypes_proto_cc",
        "@tsl//tsl/platform:strcat",
        "@xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "roofline_model_utils",
    srcs = ["roofline_model_utils.cc"],
    hdrs = ["roofline_model_utils.h"],
    deps = [
        "@xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_test(
    name = "roofline_model_utils_test",
    srcs = ["roofline_model_utils_test.cc"],
    deps = [
        ":roofline_model_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "function_registry",
    hdrs = ["function_registry.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "function_registry_test",
    srcs = ["function_registry_test.cc"],
    deps = [
        ":function_registry",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "xprof_gpu_cost_analysis_types",
    hdrs = ["xprof_gpu_cost_analysis_types.h"],
    deps = [
        ":hlo_cost_analysis_wrapper",
        "@com_google_absl//absl/strings:string_view",
        "@xla//xla/service:hlo_cost_analysis",
    ],
)

cc_library(
    name = "xla_op_utils",
    srcs = ["xla_op_utils.cc"],
    hdrs = ["xla_op_utils.h"],
    deps = [
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/parser:hlo_parser",
    ],
)

cc_test(
    name = "xla_op_utils_test",
    srcs = ["xla_op_utils_test.cc"],
    deps = [
        ":xla_op_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "xprof_gpu_cost_analysis_registration_test",
    srcs = ["xprof_gpu_cost_analysis_registration_test.cc"],
    deps = [
        "hlo_cost_analysis_wrapper",
        ":hlo_module_map",
        ":xprof_gpu_cost_analysis_registration",
        ":xprof_gpu_cost_analysis_types",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/tests:hlo_test_base",
    ],
)
