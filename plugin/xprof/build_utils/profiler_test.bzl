"""Module defining test target to run on multiple platforms"""

load("@rules_python//python:defs.bzl", "py_test")

def profiler_test(name, **kwargs):
    kwargs.pop("platforms")
    py_test(name = name, **kwargs)
