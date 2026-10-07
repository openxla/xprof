"""Tests for pytree outputs, per-leaf contracts and `make_fwd_bwd`."""

import json

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from xprof.xparity import numerical_validator
from xprof.xparity import xparity_tool

_BITWISE = numerical_validator.CONTRACT_BITWISE
_ULP = numerical_validator.CONTRACT_ULP


def _bump_one_ulp(x: np.ndarray) -> np.ndarray:
  raw = np.asarray(x, dtype=np.float32).view(np.uint32)
  return (raw + 1).view(np.float32)


def _ref_pair(x):
  x = np.asarray(x, dtype=np.float32)
  return x * 2.0, x + 1.0


def _cand_pair_second_off(x):
  out, lse = _ref_pair(x)
  return out, _bump_one_ulp(lse)


def _ref_dict(x):
  out, lse = _ref_pair(x)
  return {"out": out, "lse": lse}


def _cand_dict_lse_off(x):
  out, lse = _cand_pair_second_off(x)
  return {"out": out, "lse": lse}


def _softmax(x):
  return jax.nn.softmax(x, axis=-1)


@jax.custom_vjp
def _softmax_bad_bwd(x):
  return jax.nn.softmax(x, axis=-1)


def _softmax_bad_bwd_fwd(x):
  y = jax.nn.softmax(x, axis=-1)
  return y, y


def _softmax_bad_bwd_bwd(y, g):
  # Drops the -y * sum(g * y) term of the softmax VJP.
  return (y * g,)


_softmax_bad_bwd.defvjp(_softmax_bad_bwd_fwd, _softmax_bad_bwd_bwd)


class PytreeOutputTest(absltest.TestCase):

  def test_tuple_output_validates_every_leaf(self):
    report = numerical_validator.validate_kernels(
        _ref_pair,
        _ref_pair,
        shapes=(8, 8),
        dtype_str="float32",
        tier="fast_agent",
        contract=_BITWISE,
    )
    self.assertTrue(report.is_numerically_equivalent)
    leaves = report.batch_results[0].leaf_results
    assert leaves is not None
    self.assertEqual(list(leaves), ["[0]", "[1]"])
    self.assertEqual(leaves["[1]"].leaf_path, "[1]")

  def test_failing_leaf_is_named(self):
    report = numerical_validator.validate_kernels(
        _ref_pair,
        _cand_pair_second_off,
        shapes=(8, 8),
        dtype_str="float32",
        tier="fast_agent",
        contract=_BITWISE,
    )
    self.assertFalse(report.is_numerically_equivalent)
    first = report.batch_results[0]
    self.assertEqual(first.leaf_path, "[1]")
    assert first.leaf_results is not None
    self.assertTrue(first.leaf_results["[0]"].passed)
    self.assertFalse(first.leaf_results["[1]"].passed)
    self.assertIn("output leaf [1]", report.summary_message)

  def test_contract_by_leaf_relaxes_one_leaf(self):
    kwargs = dict(
        shapes=(8, 8), dtype_str="float32", tier="fast_agent", contract=_BITWISE
    )
    strict = numerical_validator.validate_kernels(
        _ref_dict, _cand_dict_lse_off, **kwargs
    )
    self.assertFalse(strict.is_numerically_equivalent)

    relaxed = numerical_validator.validate_kernels(
        _ref_dict,
        _cand_dict_lse_off,
        contract_by_leaf={"['lse']": _ULP},
        **kwargs,
    )
    self.assertTrue(relaxed.is_numerically_equivalent)
    self.assertEqual(relaxed.run_config["contract_by_leaf"], {"['lse']": _ULP})

  def test_unknown_leaf_raises_with_available_paths(self):
    with self.assertRaisesRegex(ValueError, r"Available leaves: \['\[0\]'"):
      numerical_validator.validate_kernels(
          _ref_pair,
          _ref_pair,
          shapes=(4, 4),
          dtype_str="float32",
          tier="fast_agent",
          contract_by_leaf={"['missing']": _ULP},
      )

  def test_invalid_leaf_contract_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown contract"):
      numerical_validator.validate_kernels(
          _ref_pair,
          _ref_pair,
          shapes=(4, 4),
          contract_by_leaf={"[0]": "exact"},
      )

  def test_structure_mismatch_raises(self):
    with self.assertRaisesRegex(ValueError, "Output structure mismatch"):
      numerical_validator.validate_kernels(
          _ref_pair,
          lambda x: _ref_pair(x)[0],
          shapes=(4, 4),
          dtype_str="float32",
          tier="fast_agent",
      )

  def test_tool_reports_structure_mismatch_as_json(self):
    payload = json.loads(
        xparity_tool.verify_numerical_parity(
            _ref_pair,
            lambda x: _ref_pair(x)[0],
            shapes=(4, 4),
            dtype_str="float32",
            tier="fast_agent",
        )
    )
    self.assertFalse(payload["is_numerically_equivalent"])
    self.assertIn(
        "Output structure mismatch", payload["shape_mismatch"]["error"]
    )

  def test_tool_serializes_leaf_results(self):
    payload = json.loads(
        xparity_tool.verify_numerical_parity(
            _ref_dict,
            _cand_dict_lse_off,
            shapes=(4, 4),
            dtype_str="float32",
            tier="fast_agent",
            contract="bitwise",
            contract_by_leaf="{\"['lse']\": 'ulp'}",
        )
    )
    self.assertTrue(payload["is_numerically_equivalent"])
    leaves = payload["batch_results"][0]["leaf_results"]
    self.assertEqual(sorted(leaves), ["['lse']", "['out']"])
    self.assertFalse(leaves["['lse']"]["bitwise"]["equal"])


class FlattenOutputFallbackTest(absltest.TestCase):

  def test_fallback_paths_match_jax_key_paths(self):
    value = {"b": (np.zeros(1), [np.ones(2)]), "a": np.zeros(3)}
    fallback = [
        path
        for path, _ in numerical_validator._flatten_output_fallback(value, "")
    ]
    with_jax = [path for path, _ in numerical_validator._flatten_output(value)]
    self.assertEqual(fallback, with_jax)
    self.assertEqual(fallback, ["['a']", "['b'][0]", "['b'][1][0]"])


class MakeFwdBwdTest(absltest.TestCase):

  def test_identical_kernels_pass_bitwise_on_out_and_vjp(self):
    fwd_bwd = numerical_validator.make_fwd_bwd(_softmax)
    report = numerical_validator.validate_kernels(
        fwd_bwd,
        fwd_bwd,
        shapes=(4, 16),
        dtype_str="float32",
        tier="fast_agent",
        contract=_BITWISE,
    )
    self.assertTrue(report.is_numerically_equivalent)
    leaves = report.batch_results[0].leaf_results
    assert leaves is not None
    self.assertEqual(list(leaves), ["['out']", "['vjp'][0]"])

  def test_wrong_backward_fails_only_on_vjp_leaf(self):
    report = numerical_validator.validate_kernels(
        numerical_validator.make_fwd_bwd(_softmax),
        numerical_validator.make_fwd_bwd(_softmax_bad_bwd),
        shapes=(4, 16),
        dtype_str="float32",
        tier="fast_agent",
    )
    self.assertFalse(report.is_numerically_equivalent)
    first = report.batch_results[0]
    assert first.leaf_results is not None
    self.assertTrue(first.leaf_results["['out']"].passed)
    self.assertFalse(first.leaf_results["['vjp'][0]"].passed)
    self.assertEqual(first.leaf_path, "['vjp'][0]")

  def test_cotangent_is_deterministic(self):
    fwd_bwd = numerical_validator.make_fwd_bwd(_softmax, cotangent_seed=7)
    x = jnp.linspace(-1.0, 1.0, 32, dtype=jnp.float32).reshape(2, 16)
    first = fwd_bwd(x)["vjp"][0]
    second = fwd_bwd(x)["vjp"][0]
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

  def test_argnums_selects_differentiated_inputs(self):
    fwd_bwd = numerical_validator.make_fwd_bwd(lambda a, b: a * b, argnums=1)
    result = fwd_bwd(jnp.ones(3), jnp.full(3, 2.0))
    self.assertLen(result["vjp"], 1)

  def test_no_float_args_raises(self):
    fwd_bwd = numerical_validator.make_fwd_bwd(lambda i: i * 2)
    with self.assertRaisesRegex(ValueError, "no floating-point"):
      fwd_bwd(jnp.arange(3))


if __name__ == "__main__":
  absltest.main()
