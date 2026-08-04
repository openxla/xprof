"""Unit tests for session path resolution."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest

from etils import epath

from xprof.profile_plugin.services.sessions import SessionResolver


class SessionResolverTest(unittest.TestCase):

  def setUp(self):
    self._td = tempfile.TemporaryDirectory()
    self.root = self._td.name
    self.resolver = SessionResolver(epath_module=epath)

  def tearDown(self):
    self._td.cleanup()

  def _touch_xplane(self, session_dir: str, host: str = 'h0'):
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, f'{host}.xplane.pb')
    with open(path, 'wb') as f:
      f.write(b'x')
    return session_dir

  def test_session_path_map(self):
    session = self._touch_xplane(os.path.join(self.root, 'my_session'))
    m = self.resolver.run_map_from_params(session_path=session, run_path=None)
    self.assertEqual(m, {'my_session': session})

  def test_run_path_lists_sessions(self):
    s1 = self._touch_xplane(os.path.join(self.root, 'runs', 'a'))
    s2 = self._touch_xplane(os.path.join(self.root, 'runs', 'b'))
    m = self.resolver.run_map_from_params(
        session_path=None, run_path=os.path.join(self.root, 'runs')
    )
    self.assertIn('a', m)
    self.assertIn('b', m)
    self.assertEqual(m['a'], s1)
    self.assertEqual(m['b'], s2)

  def test_session_path_takes_precedence_over_run_path(self):
    session = self._touch_xplane(os.path.join(self.root, 'direct'))
    run_root = os.path.join(self.root, 'runs')
    self._touch_xplane(os.path.join(run_root, 'a'))
    m = self.resolver.run_map_from_params(
        session_path=session, run_path=run_root
    )
    self.assertEqual(m, {'direct': session})

  def test_session_path_without_xplane_returns_empty(self):
    empty = os.path.join(self.root, 'empty_session')
    os.makedirs(empty)
    m = self.resolver.run_map_from_params(session_path=empty, run_path=None)
    self.assertEqual(m, {})

  def test_neither_param_returns_none(self):
    m = self.resolver.run_map_from_params(session_path=None, run_path=None)
    self.assertIsNone(m)

  def test_resolve_run_dir_from_run_map(self):
    session = self._touch_xplane(os.path.join(self.root, 's1'))
    run_map = {'s1': session}
    lock = threading.Lock()
    got = self.resolver.resolve_run_dir(
        's1', run_map, logdir=None, run_dir_cache={}, cache_lock=lock
    )
    self.assertEqual(got, session)

  def test_resolve_run_dir_missing_from_run_map_raises(self):
    lock = threading.Lock()
    with self.assertRaises(ValueError) as ctx:
      self.resolver.resolve_run_dir(
          'missing',
          {'s1': '/tmp/s1'},
          logdir=None,
          run_dir_cache={},
          cache_lock=lock,
      )
    self.assertIn('missing', str(ctx.exception))

  def test_resolve_run_dir_uses_cache(self):
    lock = threading.Lock()
    cache = {'cached_run': '/path/from/cache'}
    got = self.resolver.resolve_run_dir(
        'cached_run',
        run_map=None,
        logdir=None,
        run_dir_cache=cache,
        cache_lock=lock,
    )
    self.assertEqual(got, '/path/from/cache')

  def test_resolve_run_dir_logdir_layout(self):
    logdir = self.root
    # Frontend run "train/run1" → logdir/train/plugins/profile/run1
    expected = os.path.join(logdir, 'train', 'plugins', 'profile', 'run1')
    os.makedirs(expected)
    lock = threading.Lock()
    got = self.resolver.resolve_run_dir(
        'train/run1',
        run_map=None,
        logdir=logdir,
        run_dir_cache={},
        cache_lock=lock,
    )
    self.assertEqual(got, expected)

  def test_resolve_run_dir_root_tb_run(self):
    logdir = self.root
    # Frontend run "run1" (no tb prefix) → logdir/plugins/profile/run1
    expected = os.path.join(logdir, 'plugins', 'profile', 'run1')
    os.makedirs(expected)
    lock = threading.Lock()
    got = self.resolver.resolve_run_dir(
        'run1',
        run_map=None,
        logdir=logdir,
        run_dir_cache={},
        cache_lock=lock,
    )
    self.assertEqual(got, expected)

  def test_resolve_run_dir_empty_logdir_raises(self):
    lock = threading.Lock()
    with self.assertRaises(RuntimeError):
      self.resolver.resolve_run_dir(
          'run1',
          run_map=None,
          logdir=None,
          run_dir_cache={},
          cache_lock=lock,
      )

  def test_tb_run_directory(self):
    from xprof.profile_plugin.services.sessions import tb_run_directory

    self.assertEqual(tb_run_directory('/log', '.'), '/log')
    self.assertEqual(
        tb_run_directory('/log', 'train'), os.path.join('/log', 'train')
    )


if __name__ == '__main__':
  unittest.main()
