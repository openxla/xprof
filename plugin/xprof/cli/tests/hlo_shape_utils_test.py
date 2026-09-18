"""Unit tests for hlo_shape_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import hlo_shape_utils


class HloShapeUtilsTest(parameterized.TestCase):

  def test_parse_hlo_tensor_shapes_matmul(self):
    expr = (
        "%custom-call.1 = bf16[4,4096,4096]{2,1,0} custom-call("
        "bf16[4,4096,2048]{2,1,0} %p0, bf16[2048,4096]{1,0} %p1), "
        "custom_call_target=\"tpu_custom_call\""
    )
    out_shape, operand_shapes = hlo_shape_utils.parse_hlo_tensor_shapes(expr)
    self.assertEqual(out_shape, ("bf16", [4, 4096, 4096]))
    self.assertEqual(
        operand_shapes,
        [("bf16", [4, 4096, 2048]), ("bf16", [2048, 4096])],
    )

  def test_derive_custom_call_flops_and_bytes_contraction(self):
    expr = (
        "%custom-call.1 = bf16[4,4096,4096] custom-call("
        "bf16[4,4096,2048] %p0, bf16[2048,4096] %p1)"
    )
    flops, bytes_acc, provenance = (
        hlo_shape_utils.derive_custom_call_flops_and_bytes(
            expr, category="custom-call", name="%custom-call.1"
        )
    )
    # 2 * B * M * N * K = 2 * 4 * 4096 * 4096 * 2048 = 274877906944
    expected_flops = float(2 * 4 * 4096 * 4096 * 2048)
    # Bytes = 2 * (4*4096*4096 + 4*4096*2048 + 2048*4096)
    expected_bytes = float(
        2 * (4 * 4096 * 4096 + 4 * 4096 * 2048 + 2048 * 4096)
    )
    self.assertEqual(provenance, "derived_from_shapes")
    self.assertEqual(flops, expected_flops)
    self.assertEqual(bytes_acc, expected_bytes)

  def test_derive_custom_call_flops_and_bytes_opaque(self):
    expr = "%custom-call.2 = u32[128] custom-call(u32[128] %p0)"
    flops, bytes_acc, provenance = (
        hlo_shape_utils.derive_custom_call_flops_and_bytes(
            expr, category="custom-call", name="%custom-call.2"
        )
    )
    self.assertEqual(provenance, "opaque_custom_call")
    self.assertIsNone(flops)
    self.assertIsNone(bytes_acc)

  def test_derive_standard_xla_op(self):
    expr = "%add.1 = f32[1024,1024] add(f32[1024,1024] %a, f32[1024,1024] %b)"
    flops, bytes_acc, provenance = (
        hlo_shape_utils.derive_custom_call_flops_and_bytes(
            expr, category="elementwise", name="%add.1"
        )
    )
    self.assertEqual(provenance, "xla_cost_model")
    self.assertIsNone(flops)
    self.assertIsNone(bytes_acc)

  def test_derive_single_operand_2d_pallas_returns_opaque(self):
    expr = (
        "%run_pallas.1 = f32[2048,2048] custom-call(f32[2048,2048] %x_ref.1),"
        " custom_call_target=\"tpu_custom_call\""
    )
    flops, bytes_acc, provenance = (
        hlo_shape_utils.derive_custom_call_flops_and_bytes(
            expr, category="custom-call", name="%run_pallas.1"
        )
    )
    self.assertEqual(provenance, "opaque_custom_call")
    self.assertIsNone(flops)
    self.assertIsNone(bytes_acc)

  def test_derive_v7x_pallas_matmul_shape(self):
    expr = (
        "%matmul_optimized.1 = f32[8192,4096]{1,0:T(8,128)} custom-call("
        "f32[8192,1024]{1,0:T(8,128)} %x.1, f32[1024,4096]{1,0:T(8,128)} %y.1),"
        " custom_call_target=\"tpu_custom_call\""
    )
    flops, bytes_acc, provenance = (
        hlo_shape_utils.derive_custom_call_flops_and_bytes(
            expr, category="custom-call", name="matmul_optimized.1"
        )
    )
    self.assertEqual(provenance, "derived_from_shapes")
    self.assertEqual(flops, 68719476736.0)
    self.assertEqual(bytes_acc, 184549376.0)


if __name__ == "__main__":
  absltest.main()
