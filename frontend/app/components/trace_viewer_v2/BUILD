load("//javascript/tools/jscompiler/builddefs:flags.bzl", "RECOMMENDED_FLAGS")
load("//javascript/typescript:build_defs.bzl", "ts_declaration", "ts_development_sources", "ts_library")

# load("//third_party/bazel_rules/rules_cc/cc:cc_binary.bzl", "cc_binary")
# load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
# load("//third_party/bazel_rules/rules_cc/cc:cc_test.bzl", "cc_test")
# load("//third_party/bazel_rules/rules_wasm/wasm:defs.bzl", "wasm_cc_binary")
load("//tools/build_defs/js:rules.bzl", "js_binary")
load("//tools/build_defs/js/devserver:web_dev_server.bzl", "web_dev_server")
load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")
load(":build_defs.bzl", "wasm_cc_library", "wasm_cc_test")

package(default_visibility = [
    "//perftools/accelerators/xprof/frontend:__subpackages__",
    "//perftools/gputools/profiler/collector:__subpackages__",
    "@org_xprof//frontend:__subpackages__",
])

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
    deps = [
        "//third_party/bazel_rules/rules_cc/cc:core_rules",
        "//third_party/bazel_rules/rules_wasm/wasm:bzl_lib",
    ],
)

cc_library(
    name = "animation",
    hdrs = ["animation.h"],
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
        "//third_party/dear_imgui",
        "//util/math:mathutil",
    ],
)

cc_test(
    name = "canvas_state_test",
    srcs = ["canvas_state_test.cc"],
    deps = [
        ":canvas_state",
        "//third_party/dear_imgui",
        "@com_google_googletest//:gtest_main",
    ],
)

wasm_cc_library(
    name = "application",
    srcs = ["application.cc"],
    hdrs = ["application.h"],
    deps = [
        ":animation",
        ":canvas_state",
        ":event_manager",
        ":input_handler",
        ":webgpu_render_platform",
        "@org_xprof//frontend/app/components/trace_viewer_v2/fonts",
        "@org_xprof//frontend/app/components/trace_viewer_v2/timeline",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/time",
        "//third_party/dear_imgui",
        #data provider
        "@org_xprof//frontend/app/components/trace_viewer_v2/timeline:data_provider",
    ],
)

wasm_cc_library(
    name = "input_handler",
    srcs = ["input_handler.cc"],
    hdrs = ["input_handler.h"],
    deps = [
        "//third_party/dear_imgui",
        "//util/gtl:flat_map",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "event_data",
    hdrs = ["event_data.h"],
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
        "//third_party/emscripten:embind",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

wasm_cc_test(
    name = "event_manager_test",
    srcs = ["event_manager_test.cc"],
    linkopts = [
        "-sASYNCIFY=1",
        "-sNO_EXIT_RUNTIME=1",
    ],
    deps = [
        ":event_manager",
        "@com_google_googletest//:gtest_main",
    ],
)

wasm_cc_library(
    name = "imgui_webgpu_backend",
    srcs = ["imgui_webgpu_backend.cc"],
    hdrs = ["imgui_webgpu_backend.h"],
    deps = [
        "//third_party/dear_imgui",
        "//third_party/emdawnwebgpu:webgpu",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
    ],
)

wasm_cc_library(
    name = "webgpu_render_platform",
    srcs = ["webgpu_render_platform.cc"],
    hdrs = ["webgpu_render_platform.h"],
    deps = [
        ":canvas_state",
        ":imgui_webgpu_backend",
        "//third_party/dear_imgui",
        "//third_party/emdawnwebgpu:webgpu",
        "@com_google_absl//absl/log:check",
    ],
)

BINARY_LINKOPTS = [
    "-sASYNCIFY",
    "-sMODULARIZE",
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
        ":animation",
        ":application",
        ":canvas_state",
        ":imgui_webgpu_backend",
        ":input_handler",
        ":webgpu_render_platform",
        "//third_party/emscripten:embind",
        "@org_xprof//frontend/app/components/trace_viewer_v2/fonts",
        "@org_xprof//frontend/app/components/trace_viewer_v2/timeline",
        "@org_xprof//frontend/app/components/trace_viewer_v2/trace_helper:trace_event_parser",
    ],
)

wasm_cc_binary(
    name = "trace_viewer_v2_wasm",
    cc_target = ":trace_viewer_v2",
    generate_ts_defs = "embind",
)

ts_declaration(
    name = "wasm_declared_types",
    srcs = [":trace_viewer_v2_wasm/trace_viewer_v2.d.ts"],
    deps = ["//third_party/javascript/typings/emscripten"],
)

ts_library(
    name = "app_bundle_ts",
    srcs = ["main.ts"],
    deps = [
        ":wasm_declared_types",
        "//third_party/javascript/webgpu_types",
    ],
)

ts_development_sources(
    name = "devsrcs",
    deps = [":app_bundle_ts"],
)

JS_COMPILER_FLAGS = []

js_binary(
    name = "app_bundle",
    defs = JS_COMPILER_FLAGS + RECOMMENDED_FLAGS,
    deps = [":app_bundle_ts"],
)

filegroup(
    name = "static_files",
    srcs = [
        "index.html",
        ":app_bundle.js",
        ":trace_viewer_v2_wasm/trace_viewer_v2.js",
        ":trace_viewer_v2_wasm/trace_viewer_v2.wasm",
    ],
)

web_dev_server(
    name = "dev_server",
    concatjs_routes = {":devsrcs": "/app_bundle.js"},
    static_files = [
        "index.html",
        ":trace_viewer_v2_wasm",
    ],
    static_files_path_aliases = {
        "/trace_viewer_v2.js": "third_party/xprof/frontend/app/components/trace_viewer_v2/trace_viewer_v2_wasm/trace_viewer_v2.js",
        "/trace_viewer_v2.wasm": "third_party/xprof/frontend/app/components/trace_viewer_v2/trace_viewer_v2_wasm/trace_viewer_v2.wasm",
    },
)
