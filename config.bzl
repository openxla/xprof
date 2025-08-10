"""Configuration for the xprof repository."""

def _repository_configuration(repository_ctx):
    python_version = repository_ctx.os.environ.get("PROFILER_PYTHON_VERSION", "3.12")
    repository_ctx.file("BUILD")
    requirements_file = "//:requirements_lock_{version}.txt".format(version = python_version.replace(".", "_"))
    repository_ctx.file("repository_config.bzl", content = """
PROFILER_PYTHON_VERSION={python_version}
PROFILER_REQUIREMENTS_FILE={requirements_file}
""".format(python_version = repr(python_version), requirements_file = repr(requirements_file)))

repository_configuration = repository_rule(
    implementation = _repository_configuration,
    environ = ["PROFILER_PYTHON_VERSION"],
)
