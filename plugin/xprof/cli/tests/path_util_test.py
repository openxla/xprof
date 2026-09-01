"""Tests for path_util module."""

import getpass
import pathlib
import tempfile
from unittest import mock
from absl.testing import absltest
from xprof.cli.internal import path_util


class PathUtilTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Reset global session resolver before each test.
    path_util.register_session_resolver(lambda s: None)

  def test_get_cache_dir_format_and_creation(self):
    cache_dir = path_util.get_cache_dir()
    self.assertIsInstance(cache_dir, pathlib.Path)
    self.assertTrue(cache_dir.exists())
    self.assertIn("xprof_cli_cache_", cache_dir.name)

  def test_get_cache_dir_rejects_symlink(self):
    temp_dir = self.create_tempdir()
    target_dir = temp_dir.mkdir("target")
    link_path = pathlib.Path(temp_dir.full_path) / "xprof_cli_cache_fakeuser"
    link_path.symlink_to(target_dir.full_path)
    with mock.patch.object(
        tempfile, "gettempdir", return_value=temp_dir.full_path
    ):
      with mock.patch.object(getpass, "getuser", return_value="fakeuser"):
        with self.assertRaisesRegex(RuntimeError, "cannot be a symlink"):
          path_util.get_cache_dir()

  def test_compute_path_fingerprint_no_inputs(self):
    empty_dir = self.create_tempdir().full_path
    self.assertEqual(
        path_util.compute_path_fingerprint(empty_dir), "NO_TRACE_INPUTS"
    )
    self.assertEqual(
        path_util.compute_path_fingerprint(12345), "NO_TRACE_INPUTS"
    )
    self.assertEqual(path_util.compute_path_fingerprint(""), "NO_TRACE_INPUTS")

  def test_compute_path_fingerprint_nonexistent(self):
    self.assertEqual(
        path_util.compute_path_fingerprint(
            "/path/to/nonexistent/directory/xyz"
        ),
        "NONEXISTENT",
    )

  def test_compute_path_fingerprint_single_file(self):
    temp_dir = self.create_tempdir()
    test_file = temp_dir.create_file("test_trace.pb", content=b"sample_data")
    fp = path_util.compute_path_fingerprint(test_file.full_path)
    self.assertTrue(fp.startswith("f:"))
    self.assertLen(fp, 18)  # 'f:' + 16 hex chars

  def test_compute_path_fingerprint_directory_discovery(self):
    temp_dir = self.create_tempdir()
    temp_dir.create_file("node1.xplane.pb", content=b"trace_content_1")
    temp_dir.create_file("node2.xspace.pb", content=b"trace_content_2")
    # File that should be ignored by the fallback:
    temp_dir.create_file("ignored.op_stats_v2.pb", content=b"ignored_content")

    fp1 = path_util.compute_path_fingerprint(temp_dir.full_path)
    self.assertNotEqual(fp1, "NO_TRACE_INPUTS")
    self.assertNotEqual(fp1, "NONEXISTENT")
    self.assertLen(fp1, 16)

    # Modifying the trace file should change the fingerprint
    temp_dir.create_file("node1.xplane.pb", content=b"modified_trace_content")
    fp2 = path_util.compute_path_fingerprint(temp_dir.full_path)
    self.assertNotEqual(fp1, fp2)

  def test_compute_path_fingerprint_with_explicit_xspace_paths(self):
    temp_dir = self.create_tempdir()
    f1 = temp_dir.create_file("a.xplane.pb", content=b"a")
    f2 = temp_dir.create_file("b.xplane.pb", content=b"b")

    fp = path_util.compute_path_fingerprint(
        temp_dir.full_path, xspace_paths=[f1.full_path, f2.full_path]
    )
    self.assertNotEqual(fp, "NO_TRACE_INPUTS")
    self.assertLen(fp, 16)

  def test_register_session_resolver(self):
    temp_dir = self.create_tempdir()
    temp_dir.create_file("events.xplane.pb", content=b"session_trace_data")

    # Before registering resolver: abstract ID is non-existent
    self.assertEqual(
        path_util.compute_path_fingerprint("abstract_session_id_123"),
        "NONEXISTENT",
    )

    # Register resolver mapping abstract ID to temp_dir
    path_util.register_session_resolver(
        lambda s: pathlib.Path(temp_dir.full_path)
        if s == "abstract_session_id_123"
        else None
    )

    fp = path_util.compute_path_fingerprint("abstract_session_id_123")
    self.assertNotEqual(fp, "NONEXISTENT")
    self.assertNotEqual(fp, "NO_TRACE_INPUTS")
    self.assertLen(fp, 16)


if __name__ == "__main__":
  absltest.main()
