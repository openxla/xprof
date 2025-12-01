# Description:
#  TensorBoard plugin for the performance profiling.
package(default_visibility = ["//plugin:internal"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "profile_plugin",
    srcs = ["profile_plugin.py"],
    deps = [
        "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:plugin_asset_util",
        "@org_tensorflow_tensorboard//tensorboard/plugins:base_plugin",
        "@org_xprof//plugin/xprof/convert:all_libs",
        "@org_xprof//plugin/xprof/protobuf:protos_all_py",
        "@org_xprof//plugin/xprof/standalone:tensorboard_shim",
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
        "@pypi//etils",
        "@pypi//six",
        "@pypi//werkzeug",
    ],
)

py_library(
    name = "profile_plugin_test_utils",
    srcs = ["profile_plugin_test_utils.py"],
    deps = [
        ":profile_plugin",
        "@org_pocoo_werkzeug",
        "@org_xprof//plugin/xprof/protobuf:protos_all_py",
        "@org_xprof//plugin/xprof/standalone:tensorboard_shim",
        "@pypi//werkzeug",
    ],
)

py_test(
    name = "profile_plugin_with_tb_test",
    srcs = ["profile_plugin_test.py"],
    imports = [".."],
    main = "profile_plugin_test.py",
    deps = [
        ":profile_plugin",
        ":profile_plugin_test_utils",
        "@org_tensorflow_tensorboard//tensorboard:context",
        "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:data_provider",
        "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:event_multiplexer",
        "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:plugin_asset_util",
        "@org_tensorflow_tensorboard//tensorboard/plugins:base_plugin",
        "@org_xprof//plugin/xprof/protobuf:protos_all_py",
        "@org_xprof//plugin/xprof/standalone:tensorboard_shim",
        "@pypi//absl-py",
    ],
)

py_test(
    name = "profile_plugin_without_tb_test",
    srcs = ["profile_plugin_test.py"],
    imports = [".."],
    main = "profile_plugin_test.py",
    deps = [
        ":profile_plugin",
        ":profile_plugin_test_utils",
        "@org_xprof//plugin/xprof/protobuf:protos_all_py",
        "@org_xprof//plugin/xprof/standalone:tensorboard_shim",
        "@pypi//absl-py",
    ],
)

py_library(
    name = "profile_plugin_loader",
    srcs = ["profile_plugin_loader.py"],
    deps = [
        ":profile_plugin",
        "@org_xprof//plugin/xprof/standalone:tensorboard_shim",
    ],
)

py_library(
    name = "server",
    srcs = ["server.py"],
    deps = [
        ":profile_plugin_loader",
        "@org_xprof//plugin/xprof/standalone:tensorboard_shim",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
)

filegroup(
    name = "resources",
    srcs = [
        "__init__.py",
        "static/index.html",
        "static/index.js",
        "static/materialicons.woff2",
    ],
    visibility = ["//plugin:__subpackages__"],
)
