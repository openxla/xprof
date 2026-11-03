"""Configuration for the xprof repository."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _repository_configuration(repository_ctx):
    python_version = repository_ctx.os.environ.get("HERMETIC_PYTHON_VERSION", "3.10")
    repository_ctx.file("BUILD")
    requirements_file = "//:requirements_lock_{version}.txt".format(version = python_version.replace(".", "_"))
    repository_ctx.file("repository_config.bzl", content = """
HERMETIC_PYTHON_VERSION={python_version}
PROFILER_REQUIREMENTS_FILE={requirements_file}
""".format(python_version = repr(python_version), requirements_file = repr(requirements_file)))

repository_configuration = repository_rule(
    implementation = _repository_configuration,
    environ = ["HERMETIC_PYTHON_VERSION"],
)

_EMSCRIPTEN_BUILD_FILE_CONTENT = """
package(default_visibility = ['//visibility:public'])

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

filegroup(
    name = "includes",
    srcs = glob([
        "emscripten/cache/sysroot/include/c++/v1/**",
        "emscripten/cache/sysroot/include/compat/**",
        "emscripten/cache/sysroot/include/**",
        "lib/clang/**/include/**",
    ]),
)

filegroup(
    name = "emcc_common",
    srcs = [
        "emscripten/emcc.py",
        "emscripten/embuilder.py",
        "emscripten/emscripten-version.txt",
        "emscripten/cache/sysroot_install.stamp",
        "emscripten/src/settings.js",
        "emscripten/src/settings_internal.js",
    ] + glob(
        include = [
            "emscripten/third_party/**",
            "emscripten/tools/**",
        ],
        exclude = [
            "**/__pycache__/**",
        ],
    ),
)

filegroup(
    name = "compiler_files",
    srcs = [
        "bin/clang{bin_extension}",
        "bin/clang++{bin_extension}",
        ":emcc_common",
        ":includes",
    ],
)

filegroup(
    name = "linker_files",
    srcs = [
        "bin/clang{bin_extension}",
        "bin/llvm-ar{bin_extension}",
        "bin/llvm-dwarfdump{bin_extension}",
        "bin/llvm-nm{bin_extension}",
        "bin/llvm-objcopy{bin_extension}",
        "bin/wasm-ctor-eval{bin_extension}",
        "bin/wasm-emscripten-finalize{bin_extension}",
        "bin/wasm-ld{bin_extension}",
        "bin/wasm-metadce{bin_extension}",
        "bin/wasm-opt{bin_extension}",
        "bin/wasm-split{bin_extension}",
        "bin/wasm2js{bin_extension}",
        ":emcc_common",
    ] + glob(
        include = [
            "emscripten/cache/sysroot/lib/**",
            "emscripten/node_modules/**",
            "emscripten/src/**",
        ],
    ),
)

filegroup(
    name = "ar_files",
    srcs = [
        "bin/llvm-ar{bin_extension}",
        "emscripten/emar.py",
        "emscripten/emscripten-version.txt",
        "emscripten/src/settings.js",
        "emscripten/src/settings_internal.js",
    ] + glob(
        include = [
            "emscripten/tools/**",
        ],
        exclude = [
            "**/__pycache__/**",
        ],
    ),
)
"""

def _emsdk_ext_impl(_mctx):
    rev_hash = "14767574a5c37ff9526a253a65ddbe0811cb3667"
    emscripten_url = "https://storage.googleapis.com/webassembly/emscripten-releases-builds/{}/{}/wasm-binaries{}.{}"
    http_archive(
        name = "emscripten_bin_linux",
        build_file_content = _EMSCRIPTEN_BUILD_FILE_CONTENT.format(bin_extension = ""),
        sha256 = "27fc220a9ad98d323cad73531ff563e9838c9e1205f51ee2a5632bb4266a35d2",
        strip_prefix = "install",
        type = "tar.xz",
        url = emscripten_url.format("linux", rev_hash, "", "tar.xz"),
    )
    http_archive(
        name = "emscripten_bin_linux_arm64",
        build_file_content = _EMSCRIPTEN_BUILD_FILE_CONTENT.format(bin_extension = ""),
        strip_prefix = "install",
        type = "tar.xz",
        url = emscripten_url.format("linux", rev_hash, "-arm64", "tar.xz"),
    )
    http_archive(
        name = "emscripten_bin_mac",
        build_file_content = _EMSCRIPTEN_BUILD_FILE_CONTENT.format(bin_extension = ""),
        sha256 = "c06048915595726fc2e2da6a8db3134581a6287645fb818802a9734ff9785e77",
        strip_prefix = "install",
        type = "tar.xz",
        url = emscripten_url.format("mac", rev_hash, "", "tar.xz"),
    )
    http_archive(
        name = "emscripten_bin_mac_arm64",
        build_file_content = _EMSCRIPTEN_BUILD_FILE_CONTENT.format(bin_extension = ""),
        sha256 = "35d743453d0f91857b09f00d721037bb46753aaeae373bd7f64746338db11770",
        strip_prefix = "install",
        type = "tar.xz",
        url = emscripten_url.format("mac", rev_hash, "-arm64", "tar.xz"),
    )
    http_archive(
        name = "emscripten_bin_win",
        build_file_content = _EMSCRIPTEN_BUILD_FILE_CONTENT.format(bin_extension = ".exe"),
        sha256 = "3b576e825b26426bb72854ed98752df3fcb58cc3ab1dc116566e328b79a8abb3",
        strip_prefix = "install",
        type = "zip",
        url = emscripten_url.format("win", rev_hash, "", "zip"),
    )

emsdk_ext = module_extension(implementation = _emsdk_ext_impl)
