load("//plugin/xprof/build_utils:pytype.default.bzl", "pytype_library", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(default_visibility = ["//plugin/xprof:__subpackages__"])

licenses(["notice"])

pytype_strict_contrib_test(
    name = "attention_inputs_test",
    srcs = ["attention_inputs_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "bitwise_contract_test",
    srcs = ["bitwise_contract_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:xparity_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "compact_verdict_test",
    srcs = ["compact_verdict_test.py"],
    deps = [
        "//plugin/xprof/xparity:xparity_cli_lib",
        "//plugin/xprof/xparity:xparity_tool",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "pytree_outputs_test",
    srcs = ["pytree_outputs_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:xparity_tool",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "failure_replay_test",
    srcs = ["failure_replay_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:xparity_cli_lib",
        "//plugin/xprof/xparity:xparity_tool",
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_contrib_test(
    name = "rel_l2_test",
    srcs = ["rel_l2_test.py"],
    deps = [
        "//plugin/xprof/xparity:rel_l2",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "sqnr_test",
    srcs = ["sqnr_test.py"],
    deps = [
        "//plugin/xprof/xparity:sqnr",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "row_rel_l2_contract_test",
    srcs = ["row_rel_l2_contract_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:rel_l2",
        "//plugin/xprof/xparity:xparity_cli_lib",
        "//plugin/xprof/xparity:xparity_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "sqnr_contract_validator_test",
    srcs = ["sqnr_contract_validator_test.py"],
    deps = [
        "//plugin/xprof/xparity",
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:sqnr",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "sqnr_contract_cli_test",
    srcs = ["sqnr_contract_cli_test.py"],
    deps = [
        "//plugin/xprof/xparity:sqnr",
        "//plugin/xprof/xparity:xparity_cli_lib",
        "//plugin/xprof/xparity:xparity_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "regime_coverage_test",
    srcs = ["regime_coverage_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:xparity_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "fp4_outputs_test",
    srcs = ["fp4_outputs_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:ulp",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "numerical_generator_test",
    srcs = ["numerical_generator_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "numerical_validator_test",
    srcs = ["numerical_validator_test.py"],
    shard_count = 4,
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "xparity_cli_test",
    srcs = ["xparity_cli_test.py"],
    deps = [
        "//plugin/xprof/xparity",
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:xparity_cli_lib",
        "//plugin/xprof/xparity:xparity_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "tolerance_dilemma_test",
    srcs = ["tolerance_dilemma_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "ulp_test",
    srcs = ["ulp_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:ulp",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "tolerance_headroom_test",
    srcs = ["tolerance_headroom_test.py"],
    deps = [
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:xparity_tool",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "sqnr_checks_test",
    srcs = ["sqnr_checks_test.py"],
    shard_count = 4,
    deps = [
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:numerical_validator",
        "//plugin/xprof/xparity:sqnr",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "sqnr_head_to_head_heldout",
    testonly = True,
    srcs = ["sqnr_head_to_head_heldout.py"],
    deps = ["//plugin/xprof/xparity:sqnr"],
)

pytype_strict_library(
    name = "sqnr_head_to_head_lib",
    testonly = True,
    srcs = ["sqnr_head_to_head.py"],
    deps = [
        ":sqnr_head_to_head_heldout",
        "//plugin/xprof/xparity:numerical_generator",
        "//plugin/xprof/xparity:sqnr",
        "@absl_py//absl/flags",
        "@pypi//absl_py",
    ],
)

pytype_strict_binary(
    name = "sqnr_head_to_head",
    testonly = True,
    srcs = ["sqnr_head_to_head.py"],
    main = "sqnr_head_to_head.py",
    deps = [":sqnr_head_to_head_lib"],
)

pytype_strict_contrib_test(
    name = "sqnr_head_to_head_test",
    size = "large",
    srcs = ["sqnr_head_to_head_test.py"],
    deps = [
        ":sqnr_head_to_head_heldout",
        ":sqnr_head_to_head_lib",
        "//plugin/xprof/xparity:sqnr",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
