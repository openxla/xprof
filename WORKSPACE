workspace(name = "org_xprof")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:config.bzl", "repository_configuration")

repository_configuration(name = "repository_configuration")

load("@repository_configuration//:repository_config.bzl", "HERMETIC_PYTHON_VERSION", "PROFILER_REQUIREMENTS_FILE")

print("Using Python Version = {}".format(HERMETIC_PYTHON_VERSION))

http_archive(
    name = "curl",
    build_file = "//third_party:curl.BUILD",
    sha256 = "264537d90e58d2b09dddc50944baf3c38e7089151c8986715e2aaeaaf2b8118f",
    strip_prefix = "curl-8.11.0",
    urls = ["https://curl.se/download/curl-8.11.0.tar.gz"],
)

http_archive(
    name = "nlohmann_json",
    build_file_content = """
cc_library(
    name = "json",
    hdrs = glob(["include/nlohmann/**/*.hpp"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406",
    strip_prefix = "json-3.11.3",
    urls = ["https://github.com/nlohmann/json/archive/v3.11.3.tar.gz"],
)

http_archive(
    name = "opentelemetry-cpp",
    build_file_content = """
cc_library(
    name = "api",
    hdrs = glob(["api/include/**/*.h"]),
    includes = ["api/include"],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "b149109d5983cf8290d614654a878899a68b0c8902b64c934d06f47cd50ffe2e",
    strip_prefix = "opentelemetry-cpp-1.18.0",
    urls = ["https://github.com/open-telemetry/opentelemetry-cpp/archive/v1.18.0.tar.gz"],
)

# Apache Arrow and Parquet C++ libraries (matches google3 //third_party/arrow/METADATA commit ee4d09ebef61c663c1efbfa4c18e518a03b798be)
http_archive(
    name = "arrow",
    build_file = "//third_party:arrow.BUILD",
    sha256 = "46d72113d776592195162ebd9f0b181ed224cdc3262f78508a0e7ef72e08cf74",
    strip_prefix = "arrow-ee4d09ebef61c663c1efbfa4c18e518a03b798be",
    urls = [
        "https://github.com/apache/arrow/archive/ee4d09ebef61c663c1efbfa4c18e518a03b798be.zip",
    ],
)

http_archive(
    name = "com_github_googlecloudplatform_google_cloud_cpp",
    repo_mapping = {
        "@com_github_curl_curl": "@curl",
        "@abseil-cpp": "@com_google_absl",
    },
    sha256 = "e868bdb537121d2169fbc1ef69b81f4b4f96e97891c4567a6533d4adf62bffde",
    strip_prefix = "google-cloud-cpp-3.1.0",
    urls = [
        "http://mirror.tensorflow.org/github.com/googleapis/google-cloud-cpp/archive/v3.1.0.tar.gz",
        "https://github.com/googleapis/google-cloud-cpp/archive/v3.1.0.tar.gz",
    ],
)

# XLA uses an old (2019) version of rules_closure, while Tensorboard requires a newer (2024) version.
# rules_closure has added a number of other dependencies, which we disable so that XLA can properly initialize.
http_archive(
    name = "io_bazel_rules_closure",
    patch_args = ["-p1"],
    patches = [
        "//third_party:rules_closure.patch",
    ],
    sha256 = "d413ca7b0e95650efd87d3d030188e5666b10357b2a7e22bd14c042a3e0f6380",
    strip_prefix = "rules_closure-1f6bda75fd129c64a5cdb5535f2265de0eabe8f7",
    urls = [
        "https://github.com/bazelbuild/rules_closure/archive/1f6bda75fd129c64a5cdb5535f2265de0eabe8f7.tar.gz",  # 2024-11-26
    ],
)

http_archive(
    name = "rules_java",
    sha256 = "5449ed36d61269579dd9f4b0e532cd131840f285b389b3795ae8b4d717387dd8",
    url = "https://github.com/bazelbuild/rules_java/releases/download/8.7.0/rules_java-8.7.0.tar.gz",
)

# Toolchains for ML projects
# Details: https://github.com/google-ml-infra/rules_ml_toolchain
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "c9d0b6fd6fbd3e2a548e320890ae886198778c697186cee9956eba29fcb1d552",
    strip_prefix = "rules_ml_toolchain-f9ab31989af8be3b729eb37bf9e4833eb62ddda7",
    urls = [
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/f9ab31989af8be3b729eb37bf9e4833eb62ddda7.tar.gz",
    ],
)

http_archive(
    name = "com_googlesource_code_re2",
    repo_mapping = {"@abseil-cpp": "@com_google_absl"},
    sha256 = "87f6029d2f6de8aa023654240a03ada90e876ce9a4676e258dd01ea4c26ffd67",
    strip_prefix = "re2-2025-11-05",
    urls = ["https://github.com/google/re2/archive/2025-11-05.tar.gz"],
)

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_cuda")

load("@rules_ml_toolchain//gpu/sycl:sycl_configure.bzl", "sycl_configure")
load("@rules_ml_toolchain//gpu/sycl:sycl_init_repository.bzl", "sycl_init_repository")

http_archive(
    name = "xla",
    patch_args = ["-p1"],
    patches = ["//third_party:xla.patch"],
    sha256 = "090ecbe792c713249037b1466e2452988a8878c06f5ef93ca771b24b6b6ad7c9",
    strip_prefix = "xla-d36bdf5550766a16a3444287ac2b975a4fd65d04",
    urls = [
        "https://github.com/openxla/xla/archive/d36bdf5550766a16a3444287ac2b975a4fd65d04.zip",
    ],
)

# Initialize XLA's external dependencies (phases 4 and 3).
load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

_GRPC_PATCHES = [
    "@xla//third_party/grpc:grpc.patch",
    "//third_party:grpc.patch",
]
_GRPC_SHA256 = "41b695614b26652ff9e97ce50cfd4a6c7a3d45a9fe598d1454407746499bbf2c"
_GRPC_STRIP_PREFIX = "grpc-1.81.0"
_GRPC_URLS = ["https://github.com/grpc/grpc/archive/refs/tags/v1.81.0.tar.gz"]

http_archive(
    name = "com_github_grpc_grpc",
    patch_args = ["-p1"],
    patches = _GRPC_PATCHES,
    sha256 = _GRPC_SHA256,
    strip_prefix = _GRPC_STRIP_PREFIX,
    urls = _GRPC_URLS,
)

http_archive(
    name = "grpc",
    patch_args = ["-p1"],
    patches = _GRPC_PATCHES,
    repo_mapping = {
        "@com_github_grpc_grpc": "@grpc",
    },
    sha256 = _GRPC_SHA256,
    strip_prefix = _GRPC_STRIP_PREFIX,
    urls = _GRPC_URLS,
)

load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = HERMETIC_PYTHON_VERSION,
    requirements = {
        "3.10": "//:requirements_lock_3_10.txt",
        "3.11": "//:requirements_lock_3_11.txt",
        "3.12": "//:requirements_lock_3_12.txt",
        "3.13": "//:requirements_lock_3_13.txt",
    },
)

load("@xla//tools/toolchains/python:python_repo.bzl", "python_repository")

python_repository(name = "python_version_repo")

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@python_version_repo//:py_version.bzl", "REQUIREMENTS_WITH_LOCAL_WHEELS")
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pypi",
    experimental_requirement_cycles = {
        "fsspec": [
            "fsspec",
            "gcsfs",
        ],
    },
    requirements_lock = REQUIREMENTS_WITH_LOCAL_WHEELS,
)

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

# Initialize XLA's external dependencies (phases 2, 1, 0).
load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

load(
    "@io_bazel_rules_closure//closure:repositories.bzl",
    "rules_closure_dependencies",
    "rules_closure_toolchains",
)

rules_closure_dependencies(
    omit_bazel_skylib = True,
    omit_com_google_protobuf = True,
    omit_rules_cc = True,
    omit_rules_java = True,
    omit_rules_jvm_external = True,
    omit_rules_proto = True,
    omit_rules_python = True,
    omit_zlib = True,
)

rules_closure_toolchains()

load(
    "@xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@xla//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@xla//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

http_archive(
    name = "rules_rust",
    sha256 = "08109dccfa5bbf674ff4dba82b15d40d85b07436b02e62ab27e0b894f45bb4a3",
    strip_prefix = "rules_rust-d5ab4143245af8b33d1947813d411a6cae838409",
    urls = [
        # Master branch as of 2022-01-31
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_rust/archive/d5ab4143245af8b33d1947813d411a6cae838409.tar.gz",
        "https://github.com/bazelbuild/rules_rust/archive/d5ab4143245af8b33d1947813d411a6cae838409.tar.gz",
    ],
)

http_archive(
    name = "six_archive",
    build_file = "@absl_py//third_party:six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
)

load("@rules_java//java:rules_java_deps.bzl", "rules_java_dependencies")

rules_java_dependencies()

http_archive(
    name = "aspect_rules_js",
    sha256 = "75c25a0f15a9e4592bbda45b57aa089e4bf17f9176fd735351e8c6444df87b52",
    strip_prefix = "rules_js-2.1.0",
    url = "https://github.com/aspect-build/rules_js/releases/download/v2.1.0/rules_js-v2.1.0.tar.gz",
)

load("@aspect_rules_js//js:repositories.bzl", "rules_js_dependencies")

rules_js_dependencies()

load("@aspect_bazel_lib//lib:repositories.bzl", "aspect_bazel_lib_dependencies", "aspect_bazel_lib_register_toolchains")

aspect_bazel_lib_dependencies()

aspect_bazel_lib_register_toolchains()

http_archive(
    name = "rules_nodejs",
    sha256 = "87c6171c5be7b69538d4695d9ded29ae2626c5ed76a9adeedce37b63c73bef67",
    strip_prefix = "rules_nodejs-6.2.0",
    urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/v6.2.0/rules_nodejs-v6.2.0.tar.gz"],
)

load("@rules_nodejs//nodejs:repositories.bzl", "nodejs_register_toolchains")

nodejs_register_toolchains(
    name = "nodejs",
    node_version = "20.14.0",
)

load("@aspect_rules_js//npm:repositories.bzl", "npm_translate_lock")

npm_translate_lock(
    name = "npm",
    pnpm_lock = "//:pnpm-lock.yaml",
)

load("@npm//:repositories.bzl", "npm_repositories")

npm_repositories()



http_archive(
    name = "org_tensorflow_tensorboard",
    patch_args = ["-p1"],
    patches = ["//third_party:tensorboard.patch"],
    sha256 = "04471935801ccab0bc39951ad84aff61d829f5f5b387f0442a3a143ab58c2dbe",
    strip_prefix = "tensorboard-2.19.0",
    urls = ["https://github.com/tensorflow/tensorboard/archive/refs/tags/2.19.0.tar.gz"],
)

load("@org_tensorflow_tensorboard//third_party:js.bzl", "tensorboard_js_workspace")

tensorboard_js_workspace()

# Required by Perfetto.
http_archive(
    name = "rules_android",
    sha256 = "fe3d8c4955857b44019d83d05a0b15c2a0330a6a0aab990575bb397e9570ff1b",
    strip_prefix = "rules_android-0.6.0-alpha1",
    url = "https://github.com/bazelbuild/rules_android/releases/download/v0.6.0-alpha1/rules_android-v0.6.0-alpha1.tar.gz",
)

http_archive(
    name = "perfetto",
    sha256 = "b25023f3281165a1a7d7cde9f3ed2dfcfce022ffd727e77f6589951e0ba6af9a",
    strip_prefix = "perfetto-53.0",
    urls = ["https://github.com/google/perfetto/archive/refs/tags/v53.0.tar.gz"],
)

http_archive(
    name = "perfetto_cfg",
    build_file_content = "exports_files([\"perfetto_cfg.bzl\"])",
    sha256 = "b25023f3281165a1a7d7cde9f3ed2dfcfce022ffd727e77f6589951e0ba6af9a",
    strip_prefix = "perfetto-53.0/bazel/standalone",
    urls = ["https://github.com/google/perfetto/archive/refs/tags/v53.0.tar.gz"],
)

http_archive(
    name = "emsdk",
    # TODO(b/490301506): Remove this patch once emsdk version is upgraded
    patch_args = ["-p0"],
    patches = ["//third_party:emsdk.patch"],
    sha256 = "2d3292d508b4f5477f490b080b38a34aaefed43e85258a1de72cb8dde3f8f3af",
    strip_prefix = "emsdk-4.0.6/bazel",
    url = "https://github.com/emscripten-core/emsdk/archive/4.0.6.tar.gz",
)

load("@emsdk//:deps.bzl", emsdk_deps = "deps")

emsdk_deps()

load("@emsdk//:emscripten_deps.bzl", emsdk_emscripten_deps = "emscripten_deps")

emsdk_emscripten_deps()

http_archive(
    name = "imgui",
    build_file_content = """
licenses(["notice"])
cc_library(
    name = "imgui",
    srcs = glob(["*.cpp"], exclude=["backends/**", "misc/**"]),
    hdrs = glob(["*.h"]),
    copts = [
        "-I.",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "imgui_freetype",
    srcs = ["misc/freetype/imgui_freetype.cpp"],
    hdrs = ["misc/freetype/imgui_freetype.h"],
    copts = [
        "-I.",
        "-DFREETYPE_GLYPH_RANGES=1",
    ],
    includes = ["."],
    linkopts = ["-sUSE_FREETYPE=1"],
    visibility = ["//visibility:public"],
    deps = [":imgui"],
)
""",
    sha256 = "81087a74599e5890a07b636887cee73a7dc1a9eb9e1f19a4a0d82a76090bf4c2",
    strip_prefix = "imgui-1.88",
    urls = ["https://github.com/ocornut/imgui/archive/v1.88.zip"],
)

http_archive(
    name = "emdawnwebgpu",
    build_file_content = """
licenses(["notice"])

cc_library(
    name = "webgpu",
    srcs = glob(
        ["**/*.cpp"],
        exclude = [
            "**/android/**",
            "**/jni/**",
            "**/art/**",
            "examples/**",
            "generator/**",
            "tests/**",
            "src/dawn/tests/**",
            "src/dawn/common/IOSurfaceUtils.cpp",
            "src/dawn/utils/OSXTimer.cpp",
        ],
    ),
    hdrs = glob(["**/*.h"]),
    includes = [
        "include",
        "src",
        "src/dawn/include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
    ],
)
""",
    sha256 = "f49683605487f62e1c9e32ed0d71a3ed3029993f818f5a97f13f8266c39b0004",
    strip_prefix = "dawn-20250713.025201",
    urls = ["https://github.com/google/dawn/archive/v20250713.025201.tar.gz"],
)

load("@emsdk//:toolchains.bzl", "register_emscripten_toolchains")

# TODO(jonahweaver): Remove this once Emscripten toolchains are properly supported by Bazel.

register_emscripten_toolchains()
