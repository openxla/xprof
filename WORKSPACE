workspace(name = "org_xprof")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:config.bzl", "repository_configuration")

repository_configuration(name = "repository_configuration")

load("@repository_configuration//:repository_config.bzl", "PROFILER_PYTHON_VERSION", "PROFILER_REQUIREMENTS_FILE")

print("Using Python Version = {}".format(PROFILER_PYTHON_VERSION))

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "rules_python",
    sha256 = "690e0141724abb568267e003c7b6d9a54925df40c275a870a4d934161dc9dd53",
    strip_prefix = "rules_python-0.40.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.40.0/rules_python-0.40.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

python_register_toolchains(
    name = "python",
    ignore_root_user_error = True,
    # Available versions are listed in @rules_python//python:versions.bzl.
    # We recommend using the same version your team is already standardized on.
    python_version = PROFILER_PYTHON_VERSION,
)

load("@python//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "python_deps",
    experimental_requirement_cycles = {
        "fsspec": [
            "fsspec",
            "gcsfs",
        ],
    },
    python_interpreter_target = interpreter,
    requirements_lock = PROFILER_REQUIREMENTS_FILE,
)

load("@python_deps//:requirements.bzl", "install_deps")

install_deps()

http_archive(
    name = "io_bazel_rules_webtesting",
    sha256 = "6e104e54c283c94ae3d5c6573cf3233ce478e89e0f541a869057521966a35b8f",
    strip_prefix = "rules_webtesting-b6fc79c5a37cd18a5433fd080c9d2cc59548222c",
    urls = ["https://github.com/bazelbuild/rules_webtesting/archive/b6fc79c5a37cd18a5433fd080c9d2cc59548222c.tar.gz"],
)

http_archive(
    name = "com_google_absl",
    sha256 = "0ddd37f347c58d89f449dd189a645bfd97bcd85c5284404a3af27a3ca3476f39",
    strip_prefix = "abseil-cpp-fad946221cec37175e762c399760f54b9de9a9fa",
    url = "https://github.com/abseil/abseil-cpp/archive/fad946221cec37175e762c399760f54b9de9a9fa.tar.gz",
)

http_archive(
    name = "com_google_absl_py",
    sha256 = "a7c51b2a0aa6357a9cbb2d9437e8cd787200531867dc02565218930b6a32166e",
    strip_prefix = "abseil-py-1.0.0",
    urls = [
        "https://github.com/abseil/abseil-py/archive/refs/tags/v1.0.0.tar.gz",
    ],
)

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
    build_file = "@com_google_absl_py//third_party:six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
)

http_archive(
    name = "rules_java",
    sha256 = "a9690bc00c538246880d5c83c233e4deb83fe885f54c21bb445eb8116a180b83",
    url = "https://github.com/bazelbuild/rules_java/releases/download/7.12.2/rules_java-7.12.2.tar.gz",
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "ae060075a7c468eee42e6a08ddbb83f5a6663bdfdbd461261a465f4a3ae8598c",
    strip_prefix = "rules_closure-7f3d3351a8cc31fbaa403de7d35578683c17b447",
    urls = [
        "https://github.com/bazelbuild/rules_closure/archive/7f3d3351a8cc31fbaa403de7d35578683c17b447.tar.gz",  # 2024-03-11
    ],
)

load("@io_bazel_rules_closure//closure:repositories.bzl", "rules_closure_dependencies")

rules_closure_dependencies(
    omit_com_google_protobuf = True,
    omit_com_google_protobuf_js = True,
)

http_archive(
    name = "rules_nodejs",
    sha256 = "0c2277164b1752bb71ecfba3107f01c6a8fb02e4835a790914c71dfadcf646ba",
    urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/5.8.5/rules_nodejs-core-5.8.5.tar.gz"],
)

http_archive(
    name = "build_bazel_rules_nodejs",
    sha256 = "a1295b168f183218bc88117cf00674bcd102498f294086ff58318f830dd9d9d1",
    urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/5.8.5/rules_nodejs-5.8.5.tar.gz"],
)

http_archive(
    name = "aspect_rules_rollup",
    sha256 = "0b8ac7d97cd660eb9a275600227e9c4268f5904cba962939d1a6ce9a0a059d2e",
    strip_prefix = "rules_rollup-2.0.1",
    url = "https://github.com/aspect-build/rules_rollup/releases/download/v2.0.1/rules_rollup-v2.0.1.tar.gz",
)

load("@aspect_rules_rollup//rollup:dependencies.bzl", "rules_rollup_dependencies")

rules_rollup_dependencies()

http_archive(
    name = "aspect_rules_ts",
    sha256 = "6b15ac1c69f2c0f1282e41ab469fd63cd40eb2e2d83075e19b68a6a76669773f",
    strip_prefix = "rules_ts-3.6.0",
    url = "https://github.com/aspect-build/rules_ts/releases/download/v3.6.0/rules_ts-v3.6.0.tar.gz",
)

load("@aspect_rules_ts//ts:repositories.bzl", "rules_ts_dependencies")

rules_ts_dependencies(
    ts_version_from = "//:package.json",
)

load("@aspect_rules_js//js:repositories.bzl", "rules_js_dependencies")

rules_js_dependencies()

load("@aspect_rules_js//js:toolchains.bzl", "DEFAULT_NODE_VERSION", "rules_js_register_toolchains")

rules_js_register_toolchains(node_version = DEFAULT_NODE_VERSION)

load("@aspect_rules_js//npm:repositories.bzl", "npm_translate_lock")
load("//defs:sass_workaround.bzl", "SASS_DEPS")

npm_translate_lock(
    name = "npm",
    data = ["//:package.json"],
    npmrc = "//:.npmrc",
    pnpm_lock = "//:pnpm-lock.yaml",
    public_hoist_packages = {p: [""] for p in SASS_DEPS},
    quiet = False,
)

load("@npm//:repositories.bzl", "npm_repositories")

npm_repositories()

http_archive(
    name = "rules_sass",
    strip_prefix = "rules_sass-cc1e845339fc45d3c8390445014d5824b85a0948",
    urls = [
        "https://github.com/devversion/rules_sass/archive/cc1e845339fc45d3c8390445014d5824b85a0948.tar.gz",
    ],
)

load("@rules_sass//src/toolchain:repositories.bzl", "setup_rules_sass")

setup_rules_sass()

http_archive(
    name = "xla",
    patch_args = ["-p1"],
    patches = [
        "//third_party:xla.patch",
        "//third_party:xla_add_grpc_cares_darwin_arm64_support.patch",
    ],
    sha256 = "91fe743cc3de67fca94cc698ae86da67e7d0aff4090b86ae3193b310ca30b526",
    strip_prefix = "xla-baca92579992dc3877dc85186a84fd7d4eb55dfc",
    urls = [
        "https://github.com/openxla/xla/archive/baca92579992dc3877dc85186a84fd7d4eb55dfc.zip",
    ],
)

http_archive(
    name = "tsl",
    sha256 = "8cf1e1285c7b1843a7f5f787465c1ef80304b3400ed837870bc76d74ce04f5af",
    strip_prefix = "tsl-d71df2f7612583617d359c36243695097dd63726",
    urls = [
        "https://github.com/google/tsl/archive/d71df2f7612583617d359c36243695097dd63726.zip",
    ],
)

load("@xla//tools/toolchains/python:python_repo.bzl", "python_repository")

python_repository(name = "python_version_repo")

# Initialize XLA's external dependencies.
load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

load(
    "@xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
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
    "@xla//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
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
    name = "org_tensorflow_tensorboard",
    patch_args = ["-p1"],
    patches = ["//third_party:tensorboard.patch"],
    sha256 = "04471935801ccab0bc39951ad84aff61d829f5f5b387f0442a3a143ab58c2dbe",
    strip_prefix = "tensorboard-2.19.0",
    urls = ["https://github.com/tensorflow/tensorboard/archive/refs/tags/2.19.0.tar.gz"],
)

load("@org_tensorflow_tensorboard//third_party:js.bzl", "tensorboard_js_workspace")

tensorboard_js_workspace()
