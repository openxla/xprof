"""Module defining test target to run on multiple platforms"""

def profiler_test(name, **kwargs):
    kwargs.pop("platforms")
    native.py_test(name = name, **kwargs)
