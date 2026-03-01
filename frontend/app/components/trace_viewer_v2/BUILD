load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")
load("@npm//@bazel/concatjs:index.bzl", "ts_library")
load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load(":build_defs.bzl", "wasm_cc_library")

package(default_visibility = [
    "//plugin:__pkg__",
    "//third_party/xprof/frontend:__subpackages__",
    "//third_party/xprof/plugin:__subpackages__",
])

cc_library(
    name = "animation",
    hdrs = ["animation.h"],
    visibility = ["@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__"],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "animation_test",
    srcs = ["animation_test.cc"],
    deps = [
        ":animation",
        "@com_google_googletest//:gtest_main",
    ],
)

wasm_cc_library(
    name = "canvas_state",
    srcs = ["canvas_state.cc"],
    hdrs = ["canvas_state.h"],
    deps = [
        "@imgui",
    ],
)

cc_test(
    name = "canvas_state_test",
    srcs = ["canvas_state_test.cc"],
    deps = [
        ":canvas_state",
        "@com_google_googletest//:gtest_main",
        "@imgui",
    ],
)

wasm_cc_library(
    name = "application",
    srcs = ["application.cc"],
    hdrs = ["application.h"],
    visibility = ["@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__"],
    deps = [
        ":animation",
        ":canvas_state",
        ":event_manager",
        ":input_handler",
        ":webgpu_render_platform",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/time",
        "@imgui",
        "@org_xprof//frontend/app/components/trace_viewer_v2/fonts",
        "@org_xprof//frontend/app/components/trace_viewer_v2/timeline",
        "@org_xprof//frontend/app/components/trace_viewer_v2/timeline:data_provider",
    ],
)

wasm_cc_library(
    name = "input_handler",
    srcs = ["input_handler.cc"],
    hdrs = ["input_handler.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@imgui",
    ],
    alwayslink = 1,
)

cc_library(
    name = "event_data",
    hdrs = ["event_data.h"],
    visibility = ["@org_xprof//frontend/app/components/trace_viewer_v2:__subpackages__"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

wasm_cc_library(
    name = "event_manager",
    srcs = ["event_manager.cc"],
    hdrs = ["event_manager.h"],
    deps = [
        ":event_data",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

wasm_cc_library(
    name = "imgui_webgpu_backend",
    srcs = ["imgui_webgpu_backend.cc"],
    hdrs = ["imgui_webgpu_backend.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@emdawnwebgpu//:webgpu",
        "@imgui",
    ],
)

wasm_cc_library(
    name = "webgpu_render_platform",
    srcs = ["webgpu_render_platform.cc"],
    hdrs = ["webgpu_render_platform.h"],
    deps = [
        ":canvas_state",
        ":imgui_webgpu_backend",
        "@com_google_absl//absl/log:check",
        "@emdawnwebgpu//:webgpu",
        "@imgui",
    ],
)

BINARY_LINKOPTS = [
    "-sASYNCIFY",
    "-sMODULARIZE",
    "-sMINIMAL_RUNTIME=0",
    "-sENVIRONMENT=web",
    "-sALLOW_MEMORY_GROWTH",
    "-sEXPORT_NAME=loadWasmTraceViewerModule",
    "-sEXPORTED_RUNTIME_METHODS=callMain",
    "-sEXPORTED_FUNCTIONS=_main,_malloc,_free",
    "-sDISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR",
    "-g",
]

cc_binary(
    name = "trace_viewer_v2",
    srcs = ["main.cc"],
    linkopts = BINARY_LINKOPTS,
    tags = [
        "manual",
        "nobuilder",
        "notap",
        "requires_wasm_config",
    ],
    deps = [
        ":application",
        ":input_handler",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event_parser",
    ],
)

wasm_cc_binary(
    name = "trace_viewer_v2_wasm",
    cc_target = ":trace_viewer_v2",
)

ts_library(
    name = "app_bundle_ts",
    srcs = ["main.ts"],
    visibility = ["//frontend:__subpackages__"],
    deps = [
        "@npm//@types/emscripten",
        "@npm//@webgpu/types",
    ],
)

JS_COMPILER_FLAGS = []

rollup_bundle(
    name = "app_bundle",
    config_file = "//:rollup.config.js",
    entry_point = "main.ts",
    format = "iife",
    link_workspace_root = True,
    deps = [
        ":app_bundle_ts",
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
    ],
)

filegroup(
    name = "static_files",
    srcs = [
        "index.html",
        ":app_bundle.js",
        ":trace_viewer_v2_wasm",
    ],
)
