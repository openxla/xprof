# Description:
#  TensorBoard plugin for the performance profiling.
package(default_visibility = ["//plugin:internal"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "profile_io",
    srcs = ["profile_io.py"],
    deps = [
        "@pypi//etils",
    ],
)

py_library(
    name = "profile_plugin",
    srcs = glob(["profile_plugin/**/*.py"]),
    deps = [
        ":profile_io",
        ":version",
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

py_test(
    name = "profile_io_test",
    srcs = ["profile_io_test.py"],
    imports = [".."],
    main = "profile_io_test.py",
    deps = [
        ":profile_io",
        "@pypi//absl-py",
    ],
)

py_library(
    name = "profile_data",
    srcs = ["profile_data.py"],
    deps = [
        "@org_xprof//xprof/pywrap:_pywrap_profiler_plugin",
        "@pypi//etils",
    ],
)

py_test(
    name = "profile_data_test",
    srcs = ["profile_data_test.py"],
    imports = [".."],
    main = "profile_data_test.py",
    deps = [
        ":profile_data",
        "@pypi//absl-py",
        "@tsl//tsl/profiler/protobuf:xplane_proto_py",
    ],
)

py_test(
    name = "profile_plugin_package_test",
    srcs = glob(["profile_plugin/tests/**/*.py"]),
    imports = [".."],
    main = "profile_plugin/tests/run_all_tests.py",
    deps = [
        ":profile_plugin",
        ":profile_io",
        "@org_xprof//plugin/xprof/standalone:tensorboard_shim",
        "@pypi//absl-py",
        "@pypi//etils",
        "@pypi//werkzeug",
    ],
)

py_test(
    name = "tools_cache_test",
    srcs = ["tools_cache_test.py"],
    imports = [".."],
    main = "tools_cache_test.py",
    deps = [
        ":profile_plugin",
        ":profile_io",
        "@pypi//absl-py",
        "@pypi//etils",
    ],
)
