"Wrap the npmjs.com/sass tool for easier use in Bazel"

load("@npm//:sass/package_json.bzl", sass_bin = "bin")
load("//defs:sass_workaround.bzl", "SASS_DEPS")

# Convert sass input to output filename
def _sass_out(n):
    return n.replace(".scss", ".css")

def sass_binary(name, srcs, deps = [], **kwargs):
    """Runs SASS on the source files and output the resulting .css

    Replaces https://github.com/bazelbuild/rules_sass#sass_binary which doesn't work with rules_js.

    Args:
        name: A unique name for the terminal target
        srcs: A list of .scss sources
        deps: A list of sass dependencies
        **kwargs: Additional arguments
    """
    sass_bin.sass(
        name = name,
        srcs = srcs + deps + [
            # Workaround, see comment in sass_workaround.bzl
            "//:node_modules/" + p
            for p in SASS_DEPS
        ],
        outs = [_sass_out(src) for src in srcs] + ["%s.map" % _sass_out(src) for src in srcs],
        args = [
            "--load-path=node_modules",
            "--load-path={}/node_modules".format(native.package_name()),
        ] + [
            "$(execpath {}):{}/{}".format(src, native.package_name(), _sass_out(src))
            for src in srcs
        ],
        tags = [
            "local",
        ],
        **kwargs
    )
