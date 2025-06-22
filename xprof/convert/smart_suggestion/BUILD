package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "signal_extractor",
    srcs = ["signal_extractor.cc"],
    hdrs = ["signal_extractor.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_xprof//plugin/xprof/protobuf:overview_page_proto_cc",
    ],
)
