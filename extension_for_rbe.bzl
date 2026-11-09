"""Bzlmod module extension for RBE platform configuration."""

load("@xla//tools/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")

def _rbe_extension_impl(_ctx):
    initialize_rbe_configs()

rbe_configs = module_extension(
    implementation = _rbe_extension_impl,
)
