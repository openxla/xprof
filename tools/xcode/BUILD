# Local Xcode configuration for machines with Command Line Tools only
# (xcode-locator fails without a full Xcode.app, which makes apple_support
# fall back to macosx10.11 and break modern CLT SDKs).
#
# Point Bazel at this with:
#   --xcode_version_config=//tools/xcode:host_xcodes
#
# SDK versions must match what `xcrun --sdk macosx --show-sdk-version` reports
# (or an installed MacOSX*.sdk under CommandLineTools).

package(default_visibility = ["//visibility:public"])

xcode_version(
    name = "version_clt_15",
    version = "15.4.0.0.1.1234567890",
    aliases = [
        "15",
        "15.4",
        "15.4.0",
    ],
    default_ios_sdk_version = "17.5",
    default_macos_sdk_version = "15.5",
    default_tvos_sdk_version = "17.5",
    default_watchos_sdk_version = "10.5",
)

xcode_config(
    name = "host_xcodes",
    default = ":version_clt_15",
    versions = [":version_clt_15"],
)
