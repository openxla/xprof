load("//defs:defs.bzl", "sass_binary", "ts_library", "xprof_ng_module")

package(default_visibility = ["//frontend:internal"])

xprof_ng_module(
    name = "operations_table",
    srcs = [
        "operations_table.ts",
        "operations_table_module.ts",
    ],
    assets = [
        ":operations_table_css",
        "operations_table.ng.html",
    ],
    # strict_templates = False,
    deps = [
        ":operations_table_data_provider",
        "//:node_modules/@angular/core",
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/components/chart",
    ],
)

ts_library(
    name = "operations_table_data_provider",
    srcs = [
        "operations_table_data_provider.ts",
    ],
    deps = [
        "//frontend/app/common/constants",
        "//frontend/app/common/interfaces",
        "//frontend/app/common/interfaces:chart",
        "//frontend/app/components/chart:default_data_provider",
        "//frontend/app/components/chart:utils",
    ],
)

sass_binary(
    name = "operations_table_css",
    src = "operations_table.scss",
    # stack = False,
    sourcemap = False,
    deps = [
        "//frontend/app/styles:common",
    ],
)
