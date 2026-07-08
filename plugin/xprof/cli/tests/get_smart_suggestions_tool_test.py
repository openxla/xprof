from __future__ import annotations

import json
from unittest import mock

from absl.testing import absltest
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import smart_suggestion_tools
from xprof.cli.tools import get_smart_suggestions_tool


class GetSmartSuggestionsToolTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_cache = mock.create_autospec(
        decorators.Cache, instance=True, spec_set=True
    )
    mock_cache.get.return_value = decorators.Cache.UNKNOWN
    self.enter_context(
        mock.patch.object(
            decorators,
            'get_cache',
            return_value=mock_cache,
            autospec=True,
        )
    )

  @mock.patch.object(
      smart_suggestion_tools, 'fetch_smart_suggestions', autospec=True
  )
  def test_get_smart_suggestions_success(self, mock_fetch):
    expected_data = {
        'suggestions': [
            {'ruleName': 'TestRule', 'suggestionText': 'Improve things'}
        ]
    }
    mock_fetch.return_value = expected_data

    result_str = get_smart_suggestions_tool.get_smart_suggestions('dummy_sid')

    self.assertEqual(json.loads(result_str), expected_data)
    mock_fetch.assert_called_once_with('dummy_sid', strip_html=True)

  @mock.patch.object(
      smart_suggestion_tools, 'fetch_smart_suggestions', autospec=True
  )
  def test_get_smart_suggestions_rpc_failure(self, mock_fetch):
    expected_data = {
        'error': 'RPC error (session_id: dummy_sid): RPCException("Error")',
        'traceback': 'Traceback...',
    }
    mock_fetch.return_value = expected_data

    result_str = get_smart_suggestions_tool.get_smart_suggestions('dummy_sid')

    self.assertEqual(json.loads(result_str), expected_data)
    mock_fetch.assert_called_once_with('dummy_sid', strip_html=True)

  @mock.patch.object(
      smart_suggestion_tools, 'fetch_smart_suggestions', autospec=True
  )
  def test_get_smart_suggestions_http_error(self, mock_fetch):
    expected_data = {
        'error': (
            'HTTP Error 500 (session_id: dummy_sid): Internal Server Error'
        ),
        'traceback': '',
    }
    mock_fetch.return_value = expected_data

    result_str = get_smart_suggestions_tool.get_smart_suggestions('dummy_sid')

    self.assertEqual(json.loads(result_str), expected_data)
    mock_fetch.assert_called_once_with('dummy_sid', strip_html=True)


if __name__ == '__main__':
  absltest.main()
