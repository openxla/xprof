"""Xparity: Numerical accuracy and parity verification engine for ML kernels."""

from xprof.xparity import numerical_generator
from xprof.xparity import numerical_validator
from xprof.xparity import rel_l2
from xprof.xparity import ulp
from xprof.xparity import xparity_tool

ORACLE_AUTO = numerical_validator.ORACLE_AUTO
CONTRACT_ULP = numerical_validator.CONTRACT_ULP
CONTRACT_BITWISE = numerical_validator.CONTRACT_BITWISE
CONTRACT_ROW_REL_L2 = numerical_validator.CONTRACT_ROW_REL_L2
validate_kernels = numerical_validator.validate_kernels
compare_bitwise = numerical_validator.compare_bitwise
make_fwd_bwd = numerical_validator.make_fwd_bwd
replay_failure_dump = numerical_validator.replay_failure_dump
chunk_callable = numerical_validator.chunk_callable
compare_row_rel_l2 = rel_l2.compare_row_rel_l2
compute_ulp_distance = ulp.compute_ulp_distance
get_contract = ulp.get_contract
generate_test_suite = numerical_generator.generate_test_suite
generate_attention_suite = numerical_generator.generate_attention_suite
save_test_suite = numerical_generator.save_test_suite
load_test_suite = numerical_generator.load_test_suite
read_suite_metadata = numerical_generator.read_suite_metadata
verify_numerical_parity = xparity_tool.verify_numerical_parity
generate_suite = xparity_tool.generate_suite
inspect_suite = xparity_tool.inspect_suite
probe_precision = xparity_tool.probe_precision
replay = xparity_tool.replay

__all__ = [
    "CONTRACT_BITWISE",
    "CONTRACT_ROW_REL_L2",
    "CONTRACT_ULP",
    "ORACLE_AUTO",
    "chunk_callable",
    "compare_bitwise",
    "compare_row_rel_l2",
    "compute_ulp_distance",
    "generate_attention_suite",
    "generate_suite",
    "generate_test_suite",
    "get_contract",
    "inspect_suite",
    "load_test_suite",
    "make_fwd_bwd",
    "numerical_generator",
    "numerical_validator",
    "probe_precision",
    "read_suite_metadata",
    "rel_l2",
    "replay",
    "replay_failure_dump",
    "save_test_suite",
    "ulp",
    "validate_kernels",
    "verify_numerical_parity",
    "xparity_tool",
]
