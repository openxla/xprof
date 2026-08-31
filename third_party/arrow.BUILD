# Description:
#   Apache Arrow and Parquet C++ libraries.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

genrule(
    name = "config_h",
    srcs = ["cpp/src/arrow/util/config.h.cmake"],
    outs = ["cpp/src/arrow/util/config.h"],
    cmd = ("sed " +
           "-e 's/@ARROW_VERSION_MAJOR@/21/g' " +
           "-e 's/@ARROW_VERSION_MINOR@/0/g' " +
           "-e 's/@ARROW_VERSION_PATCH@/0/g' " +
           "-e 's/cmakedefine/define/g' " +
           "$< >$@"),
)

cc_library(
    name = "config",
    hdrs = ["cpp/src/arrow/util/config.h"],
    includes = ["."],
    visibility = ["//visibility:private"],
)

fbs_headers = [
    "cpp/src/generated/feather_generated.h",
    "cpp/src/generated/File_generated.h",
    "cpp/src/generated/Message_generated.h",
    "cpp/src/generated/Schema_generated.h",
    "cpp/src/generated/SparseTensor_generated.h",
    "cpp/src/generated/Tensor_generated.h",
]

cc_library(
    name = "arrow",
    srcs = glob(
        [
            "cpp/src/arrow/*.cc",
            "cpp/src/arrow/c/*.cc",
            "cpp/src/arrow/compute/**/*.cc",
            "cpp/src/arrow/array/**/*.cc",
            "cpp/src/arrow/util/**/*.cc",
            "cpp/src/arrow/util/**/*.h",
            "cpp/src/arrow/vendored/**/*.cpp",
            "cpp/src/arrow/vendored/**/*.hpp",
            "cpp/src/arrow/vendored/**/*.cc",
            "cpp/src/arrow/vendored/**/*.c",
            "cpp/src/arrow/vendored/**/*.h",
            "cpp/src/arrow/io/*.h",
            "cpp/src/arrow/io/*.cc",
            "cpp/src/arrow/tensor/*.cc",
            "cpp/src/arrow/ipc/*.h",
            "cpp/src/arrow/ipc/*.cc",
            "cpp/src/parquet/**/*.cc",
            "cpp/src/parquet/**/*.h",
        ],
        exclude = [
            "cpp/src/arrow/util/bpacking_avx2.*",
            "cpp/src/arrow/util/bpacking_avx512.*",
            "cpp/src/arrow/util/bpacking_neon*",
            "cpp/src/arrow/util/bpacking_simd*",
            "cpp/src/arrow/util/compression_brotli*",
            "cpp/src/arrow/util/compression_bz2*",
            "cpp/src/arrow/util/compression_lz4*",
            "cpp/src/arrow/util/compression_zlib*",
            "cpp/src/arrow/util/compression_snappy*",
            "cpp/src/arrow/**/*test*",
            "cpp/src/arrow/**/*benchmark*.cc",
            "cpp/src/arrow/**/*hdfs*.cc",
            "cpp/src/arrow/**/*hdfs*.h",
            "cpp/src/arrow/ipc/json*.cc",
            "cpp/src/arrow/ipc/generate*.cc",
            "cpp/src/arrow/ipc/stream-to-file.cc",
            "cpp/src/arrow/ipc/file-to-stream.cc",
            "cpp/src/arrow/vendored/xxhash/**",
            "cpp/src/arrow/vendored/datetime/**",
            "cpp/src/parquet/**/*test*",
            "cpp/src/parquet/**/*benchmark*",
        ],
    ),
    hdrs = glob([
        "cpp/src/arrow/*.h",
        "cpp/src/arrow/array/*.h",
        "cpp/src/arrow/c/*.h",
        "cpp/src/arrow/compute/**/*.h",
        "cpp/src/arrow/io/*.h",
        "cpp/src/arrow/ipc/*.h",
        "cpp/src/arrow/tensor/*.h",
        "cpp/src/arrow/util/**/*.h",
        "cpp/src/parquet/**/*.h",
    ]) + fbs_headers,
    defines = [
        "ARROW_STATIC",
        "ARROW_WITH_ZSTD",
        "PARQUET_STATIC",
    ],
    includes = [
        "cpp/src",
        "cpp/thirdparty/flatbuffers/include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":datetime",
        ":flatbuffers",
        ":xxhash",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@net_zstd//:zstd",
    ],
)

cc_library(
    name = "xxhash",
    srcs = [],
    hdrs = [
        "cpp/src/arrow/vendored/xxhash/xxhash.c",
        "cpp/src/arrow/vendored/xxhash/xxhash.h",
    ],
    copts = ["-Wno-implicit-fallthrough"],
    includes = ["."],
    visibility = ["//visibility:private"],
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "datetime",
    srcs = [
        "cpp/src/arrow/vendored/datetime/tz.cpp",
    ],
    hdrs = [
        "cpp/src/arrow/vendored/datetime/date.h",
        "cpp/src/arrow/vendored/datetime/ios.h",
        "cpp/src/arrow/vendored/datetime/tz.h",
        "cpp/src/arrow/vendored/datetime/tz_private.h",
        "cpp/src/arrow/vendored/datetime/visibility.h",
    ],
    includes = ["."],
    linkopts = select({
        ":windows": [
            "ole32.lib",  # for CoTaskMemFree
            "shell32.lib",  # for SHGetKnownFolderPath
        ],
        "//conditions:default": [
        ],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "flatbuffers",
    srcs = [],
    hdrs = [
        "cpp/thirdparty/flatbuffers/include/flatbuffers/base.h",
        "cpp/thirdparty/flatbuffers/include/flatbuffers/flatbuffers.h",
        "cpp/thirdparty/flatbuffers/include/flatbuffers/stl_emulation.h",
    ],
    copts = ["-Wno-implicit-fallthrough"],
    includes = ["."],
    visibility = ["//visibility:private"],
)

alias(
    name = "arrow_core",
    actual = ":arrow",
    visibility = ["//visibility:public"],
)

alias(
    name = "array_hdrs",
    actual = ":arrow",
    visibility = ["//visibility:public"],
)

alias(
    name = "util_hdrs",
    actual = ":arrow",
    visibility = ["//visibility:public"],
)

alias(
    name = "parquet",
    actual = ":arrow",
    visibility = ["//visibility:public"],
)

alias(
    name = "parquet_cpp2",
    actual = ":arrow",
    visibility = ["//visibility:public"],
)
