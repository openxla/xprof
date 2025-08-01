# Description:
#  Demos for profiler plugin.

load("@rules_python//python:defs.bzl", "py_binary")

package(
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "profile_demo",
    srcs = [
        "profile_demo.py",
        "profile_demo_data.py",
    ],
    data = [
        "data/profile_demo.input_pipeline.json",
        "data/profile_demo.input_pipeline.pb",
        "data/profile_demo.memory_viewer.json",
        "data/profile_demo.op_profile.json",
        "data/profile_demo.overview_page.json",
        "data/profile_demo.overview_page.pb",
        "data/profile_demo.pod_viewer.json",
        "data/profile_demo.tensorflow_stats.pb",
    ],
    deps = [
        "//:expect_tensorflow_installed",
        "//net/proto2/python/public",
        "//third_party/py/absl:app",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:plugin_asset_util",
        "@org_xprof//plugin/xprof:profile_plugin",
        "@org_xprof//plugin/xprof/protobuf:trace_events_proto_py",
        "@pypi//absl-py",  # build_cleaner: keep; go/disable_tf2
    ],
)
