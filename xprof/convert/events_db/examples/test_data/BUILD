package(
    # copybara:uncomment default_applicable_licenses = ["//xprof:license"],
    default_visibility = ["@org_xprof//xprof/convert/events_db:__subpackages__"],
    licenses = ["notice"],
)

filegroup(
    name = "test_data",
    srcs = [
        "test.xplane.pb",
        "test.xplane.pbtxt",
    ],
)

exports_files([
    "test.xplane.pb",
    "test.xplane.pbtxt",
])
