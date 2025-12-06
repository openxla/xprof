from absl.testing import absltest
from absl.testing import parameterized
from xprof.standalone import plugin_asset_util


class PluginAssetUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="No_Scheme",
          scheme="",
          expected="file",
      ),
      dict(
          testcase_name="GCS_Scheme",
          scheme="gs",
          expected="gs",
      ),
      dict(
          testcase_name="Windows_Path_C_Drive",
          scheme="C",
          expected="file",
      ),
      dict(
          testcase_name="Windows_path_T_Drive",
          scheme="T",
          expected="file",
      ),
  )
  def test_get_protocol_from_parsed_url_scheme(self, scheme, expected):
    got = plugin_asset_util._get_protocol_from_parsed_url_scheme(scheme=scheme)
    self.assertEqual(expected, got)


if __name__ == "__main__":
  absltest.main()
