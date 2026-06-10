import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client
from xprof.cli.tools import get_utilization_viewer_tool


class GetUtilizationViewerToolTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(
        decorators.Cache, instance=True, spec_set=True
    )
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(
            decorators,
            "get_cache",
            return_value=mock_cache,
            autospec=True,
        )
    )
    self.mock_client = mock.create_autospec(xprof_client.CachedXprofClient)
    self.enter_context(
        mock.patch.object(
            xprof_client,
            "get_client",
            return_value=self.mock_client,
            autospec=True,
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="success",
          csv_data="""Host,Device,Sample,Node,Name,Achieved,Peak,Unit
0,0,0,0,Vector ALUs,1.0,2.0,instructions
0,0,0,0,Scalar Unit,2.0,4.0,instructions
0,0,0,0,Vmem/Cmem Stores,3.0,6.0,instructions
0,0,0,0,Vmem Loads,4.0,8.0,instructions
0,0,0,0,Cmem Loads,5.0,10.0,instructions
0,0,0,0,MXU0,6.0,12.0,instructions
0,0,0,0,XLU0,7.0,14.0,instructions
0,0,0,0,HBM Rd+Wr,8.0,16.0,bytes
0,0,0,0,ICI (Read),9.0,18.0,bytes
0,0,0,0,ICI (Write),10.0,20.0,bytes
0,0,0,0,MXU_BF16,2.0,5.0,instructions
0,0,0,0,MXU_I8,3.0,10.0,instructions
0,0,0,0,Avg MXU Busy,4.0,10.0,instructions
""",
          kwargs={},
          expected_result={
              "vector_alu_utilization_percent": 50.0,
              "scalar_unit_utilization_percent": 50.0,
              "vmem_cmem_stores_utilization_percent": 50.0,
              "vmem_loads_utilization_percent": 50.0,
              "cmem_loads_utilization_percent": 50.0,
              "hbm_bandwidth_utilization_percent": 50.0,
              "ici_read_utilization_percent": 50.0,
              "ici_write_utilization_percent": 50.0,
              "xlu_utilization_percent": 50.0,
              "mxu_utilization_percent": 40.0,
              "idleness_percent": 50.0,
              "metrics": {
                  "Vector ALUs": 50.0,
                  "Scalar Unit": 50.0,
                  "Vmem/Cmem Stores": 50.0,
                  "Vmem Loads": 50.0,
                  "Cmem Loads": 50.0,
                  "MXU0": 50.0,
                  "XLU0": 50.0,
                  "HBM Rd+Wr": 50.0,
                  "ICI (Read)": 50.0,
                  "ICI (Write)": 50.0,
                  "MXU_BF16": 40.0,
                  "MXU_I8": 30.0,
                  "Avg MXU Busy": 40.0,
              },
          },
      ),
      dict(
          testcase_name="no_data",
          csv_data=None,
          kwargs={},
          expected_result={
              "error": "No data returned for session test-session"
          },
      ),
      dict(
          testcase_name="no_node0_data",
          csv_data="""Host,Device,Sample,Node,Name,Achieved,Peak,Unit
1,0,0,0,Vector ALUs,1.0,2.0,instructions
""",
          kwargs={},
          expected_result={
              "message": "No data found for Host 0 Device 0 Node 0"
          },
      ),
      dict(
          testcase_name="success_non_zero_parameters",
          csv_data="""Host,Device,Sample,Node,Name,Achieved,Peak,Unit
1,2,0,3,Vector ALUs,1.0,2.0,instructions
1,2,0,3,Scalar Unit,2.0,4.0,instructions
1,2,0,3,MXU0,6.0,12.0,instructions
""",
          kwargs={"host": 1, "device": 2, "node": 3},
          expected_result={
              "vector_alu_utilization_percent": 50.0,
              "scalar_unit_utilization_percent": 50.0,
              "mxu_utilization_percent": 50.0,
              "idleness_percent": 50.0,
              "metrics": {
                  "Vector ALUs": 50.0,
                  "Scalar Unit": 50.0,
                  "MXU0": 50.0,
              },
          },
      ),
      dict(
          testcase_name="missing_parameters",
          csv_data="""Host,Device,Sample,Node,Name,Achieved,Peak,Unit
1,2,0,3,Vector ALUs,1.0,2.0,instructions
""",
          kwargs={"host": 1, "device": 2, "node": 4},
          expected_result={
              "message": "No data found for Host 1 Device 2 Node 4"
          },
      ),
      dict(
          testcase_name="type_coercion",
          csv_data="""Host,Device,Sample,Node,Name,Achieved,Peak,Unit
1,2,0,3,Vector ALUs,1.0,2.0,instructions
1,2,0,3,Scalar Unit,2.0,4.0,instructions
1,2,0,3,MXU0,6.0,12.0,instructions
""",
          kwargs={"host": "1", "device": "2", "node": "3"},
          expected_result={
              "vector_alu_utilization_percent": 50.0,
              "scalar_unit_utilization_percent": 50.0,
              "mxu_utilization_percent": 50.0,
              "idleness_percent": 50.0,
              "metrics": {
                  "Vector ALUs": 50.0,
                  "Scalar Unit": 50.0,
                  "MXU0": 50.0,
              },
          },
      ),
      dict(
          testcase_name="with_no_mxu_busy",
          csv_data="""Host,Device,Sample,Node,Name,Achieved,Peak,Unit
0,0,0,0,No MXU Busy,30.0,100.0,instructions
""",
          kwargs={},
          expected_result={
              "idleness_percent": 30.0,
              "metrics": {
                  "No MXU Busy": 30.0,
              },
          },
      ),
      dict(
          testcase_name="missing_node_column_warning",
          csv_data="""Host,Device,Sample,Name,Achieved,Peak,Unit
0,0,0,Vector ALUs,1.0,2.0,instructions
""",
          kwargs={"node": 1},
          expected_result={
              "vector_alu_utilization_percent": 50.0,
              "idleness_percent": 100.0,
              "metrics": {
                  "Vector ALUs": 50.0,
              },
              "warnings": [
                  "Node column missing; ignoring node=1 filter"
              ]
          },
      ),
  )
  def test_get_utilization_viewer(self, csv_data, kwargs, expected_result):
    if csv_data is None:
      self.mock_client.fetch.return_value = (None, b"")
    else:
      self.mock_client.fetch.return_value = (None, csv_data.encode("utf-8"))

    result_str = get_utilization_viewer_tool.get_utilization_viewer(
        "test-session", **kwargs
    )
    result = json.loads(result_str)

    self.assertEqual(result, expected_result)


if __name__ == "__main__":
  absltest.main()
