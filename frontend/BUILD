load("@npm//:@angular/cli/package_json.bzl", "bin")
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = [
    "//frontend:__subpackages__",
    "//plugin:__subpackages__",
])

licenses(["notice"])

package_group(
    name = "internal",
    packages = ["//..."],
)

# Runs the Angular CLI build.
# Outputs files into dist/xprof directory.
# TODO: Migrate away from legacy Webpack builder (bin.ng) to the
# new esbuild/Vite-based application builder. Webpack-based builders are
# officially deprecated since Angular 17 and completely removed in Angular 21+.
bin.ng(
    name = "build",
    srcs = [
        "angular.json",
        "tsconfig.app.json",
        "//:node_modules",
        "//:package_json",
        "//:tsconfig",
        "//frontend/app",
        "//plugin/xprof:resources",
    ] + glob([
        "main.ts",
        "index.html",
        "styles.scss",
    ]),
    args = [
        "build",
        "--configuration",
        "production",
    ],
    chdir = "frontend",
    out_dirs = [
        "dist/xprof",
    ],
)

# Expose bundle.js from Angular CLI output
genrule(
    name = "bundle",
    srcs = [":build"],
    outs = ["bundle.js"],
    cmd = "cp $(execpath :build)/main.js $@",
)

# Expose styles.css from Angular CLI output
genrule(
    name = "css",
    srcs = [":build"],
    outs = ["styles.css"],
    cmd = "cp $(execpath :build)/styles.css $@",
)

# Expose zone.js from Angular CLI output (polyfills.js)
genrule(
    name = "zone.js",
    srcs = [":build"],
    outs = ["zone.js"],
    cmd = "cp $(execpath :build)/polyfills.js $@",
)

# Expose runtime.js from Angular CLI output
genrule(
    name = "runtime.js",
    srcs = [":build"],
    outs = ["runtime.js"],
    cmd = "cp $(execpath :build)/runtime.js $@",
)

py_binary(
    name = "server",
    testonly = True,
    srcs = ["server.py"],
    data = [
        "index.html",
        ":bundle",
        ":css",
        ":zone.js",
        ":runtime.js",
    ],
    python_version = "PY3",
)

# To run this local development server:
# 1. Start the python backend server (e.g. on port 8791):
#    /tmp/xprof_venv_3_13/bin/xprof server --logdir=/path/to/logdir --port=8791
# 2. Run this target: bazel run //frontend:serve
# 3. Open browser at http://localhost:4200/ (with HMR and Host Header bypass active)
bin.ng_binary(
    name = "serve",
    args = [
        "serve",
        "--host=0.0.0.0",
        "--port=4200",
        "--proxy-config=proxy.conf.json",
        "--disable-host-check",
    ],
    chdir = "frontend",
    data = [
        "angular.json",
        "tsconfig.app.json",
        "//:node_modules",
        "//:package_json",
        "//:tsconfig",
        "//frontend/app:app",
        "//plugin/xprof:resources",
        "proxy.conf.json",
    ] + glob([
        "main.ts",
        "index.html",
        "styles.scss",
    ]),
)
