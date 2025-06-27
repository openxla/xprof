"Wrap the npmjs.com/sass tool to be manually used for compiling material"

load("@npm//:sass/package_json.bzl", sass_bin = "bin")
load("//defs:sass_workaround.bzl", "SASS_DEPS")

# Convert sass input to output filename
def _sass_out(n):
    return n.replace(".scss", ".css")

def sass_workaround(name, src, deps = [], **kwargs):
    """Runs SASS on the source files and output the resulting .css

    Args:
        name: A unique name for the terminal target
        src: A .scss source
        deps: A list of sass dependencies
        **kwargs: Additional arguments
    """
    if "sourcemap" in kwargs:
        kwargs.pop("sourcemap")
    sass_bin.sass(
        name = name,
        srcs = [src] + deps + [
            # Workaround, see comment in sass_workaround.bzl
            "//:node_modules/" + p
            for p in SASS_DEPS
        ],
        outs = [_sass_out(src)] + ["%s.map" % _sass_out(src)],
        args = [
            "--load-path=node_modules",
            "--load-path={}/node_modules".format(native.package_name()),
        ] + [
            "$(execpath {}):{}/{}".format(src, native.package_name(), _sass_out(src)),
        ],
        **kwargs
    )
