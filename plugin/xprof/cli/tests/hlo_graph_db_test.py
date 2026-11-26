"""Unit tests for hlo_graph_db."""

import textwrap

from absl.testing import absltest
from xprof.cli.internal import hlo_graph_db

# A module with an entry computation and one fused (non-entry) computation.
_HLO_TEXT = textwrap.dedent("""\
    HloModule m

    %fused (param_0: f32[8]) -> f32[8] {
      %param_0 = f32[8]{0} parameter(0)
      ROOT %add = f32[8]{0} add(%param_0, %param_0)
    }

    ENTRY %main (arg0: f32[8]) -> f32[8] {
      %arg0 = f32[8]{0} parameter(0)
      ROOT %fusion = f32[8]{0} fusion(%arg0), kind=kLoop, calls=%fused
    }
    """)


class HloGraphDbTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.db = hlo_graph_db.HloGraphDb.from_text(_HLO_TEXT)
    self.addCleanup(self.db.close)

  def test_ensure_all_indexed_indexes_non_entry_computation(self):
    self.assertCountEqual(self.db.comp_instr_ids, ["main"])

    self.db.ensure_all_indexed()

    self.assertCountEqual(self.db.comp_instr_ids, ["fused", "main"])

  def test_ensure_all_indexed_maps_non_entry_instructions(self):
    self.db.ensure_all_indexed()

    fused_ids = self.db.comp_instr_ids["fused"]
    self.assertEqual(
        [self.db.id_to_name[iid] for iid in fused_ids], ["param_0", "add"]
    )
    self.assertEqual(self.db.id_to_name[self.db.comp_root_id["fused"]], "add")

  def test_ensure_all_indexed_second_call_changes_nothing(self):
    self.db.ensure_all_indexed()
    names_after_first_call = dict(self.db.id_to_name)
    self.assertCountEqual(
        names_after_first_call.values(), ["arg0", "fusion", "param_0", "add"]
    )

    self.db.ensure_all_indexed()

    self.assertEqual(self.db.id_to_name, names_after_first_call)


if __name__ == "__main__":
  absltest.main()
