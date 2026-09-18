"""Xparity: Numerical accuracy and parity verification engine for ML kernels."""

from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator
from xprof.xparity import xparity_tool

ORACLE_AUTO = numerical_validator.ORACLE_AUTO
validate_kernels = numerical_validator.validate_kernels
chunk_callable = numerical_validator.chunk_callable
compute_ulp_distance = numerical_validator.compute_ulp_distance
generate_test_suite = numerical_generator.generate_test_suite
save_test_suite = numerical_generator.save_test_suite
load_test_suite = numerical_generator.load_test_suite
verify_numerical_parity = xparity_tool.verify_numerical_parity
generate_suite = xparity_tool.generate_suite
inspect_suite = xparity_tool.inspect_suite
probe_precision = xparity_tool.probe_precision

__all__ = [
    "ORACLE_AUTO",
    "chunk_callable",
    "compute_ulp_distance",
    "generate_suite",
    "generate_test_suite",
    "inspect_suite",
    "load_test_suite",
    "numerical_generator",
    "numerical_validator",
    "probe_precision",
    "save_test_suite",
    "validate_kernels",
    "verify_numerical_parity",
    "xparity_tool",
]
