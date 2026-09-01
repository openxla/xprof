"""Path fingerprinting, trace discovery, and cache directory utilities."""

from collections.abc import Callable, Sequence
import getpass
import hashlib
import os
import pathlib
import tempfile
from typing import Any

# Global registry for session path resolution callbacks.
_SESSION_RESOLVER: Callable[[str], pathlib.Path | None] | None = None


def register_session_resolver(
    resolver: Callable[[str], pathlib.Path | None],
) -> None:
  """Registers a callback for resolving session IDs to directory paths."""
  global _SESSION_RESOLVER
  _SESSION_RESOLVER = resolver


def get_cache_dir() -> pathlib.Path:
  """Returns a user-specific temporary directory for the cache."""
  user = getpass.getuser()
  cache_dir = pathlib.Path(tempfile.gettempdir()) / f"xprof_cli_cache_{user}"
  if cache_dir.is_symlink():
    raise RuntimeError(f"Cache directory cannot be a symlink: {cache_dir}")
  if cache_dir.exists():
    st = cache_dir.stat()
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
      raise RuntimeError(
          f"Cache directory ownership mismatch: {cache_dir} owned by"
          f" {st.st_uid} != {os.getuid()}"
      )
    if hasattr(os, "chmod") and (st.st_mode & 0o777) != 0o700:
      try:
        os.chmod(cache_dir, 0o700)
      except OSError:
        pass
  else:
    cache_dir.mkdir(mode=0o700, exist_ok=True)
  return cache_dir


def _resolve_session_path(val: str) -> pathlib.Path | None:
  """Resolves a session ID to an on-disk directory path via resolver."""
  if _SESSION_RESOLVER is not None:
    try:
      resolved = _SESSION_RESOLVER(val)
      if resolved and pathlib.Path(resolved).exists():
        return pathlib.Path(resolved)
    except Exception:  # pylint: disable=broad-except
      pass
  return None


def compute_path_fingerprint(
    val: Any, xspace_paths: Sequence[str] | None = None
) -> str:
  """Computes a content-addressable SHA-256 fingerprint of raw trace inputs."""
  if not isinstance(val, (str, pathlib.Path)) and not xspace_paths:
    return "NO_TRACE_INPUTS"

  files: list[pathlib.Path] = []
  if xspace_paths:
    for p in xspace_paths:
      try:
        path_obj = pathlib.Path(p)
        if path_obj.is_file():
          files.append(path_obj)
      except (OSError, ValueError):
        continue
  else:
    if not val:
      return "NO_TRACE_INPUTS"
    try:
      p = pathlib.Path(val).expanduser()
      if not p.exists():
        resolved_p = _resolve_session_path(str(val))
        if resolved_p is not None:
          p = resolved_p
      if not p.exists():
        return "NONEXISTENT"
      if p.is_file():
        st = p.stat()
        hasher = hashlib.sha256()
        hasher.update(f"{p.name}:{st.st_size}:".encode("utf-8"))
        try:
          with open(p, "rb") as fp:
            hasher.update(fp.read(64 * 1024))
            if st.st_size > 128 * 1024:
              fp.seek(-64 * 1024, 2)
              hasher.update(fp.read(64 * 1024))
        except (OSError, ValueError):
          pass
        return f"f:{hasher.hexdigest()[:16]}"
    except (OSError, ValueError):
      return "NONEXISTENT"

    # Recursive trace discovery matching get_xspace_paths
    try:
      files = sorted(p.glob("**/*.xplane.pb")) + sorted(
          p.glob("**/*.xspace.pb")
      )
      if not files:
        # Fallback for generic naming, excluding generated artifacts
        files = [
            f
            for f in sorted(p.glob("**/*.pb"))
            if not f.name.endswith(".op_stats_v2.pb")
            and not f.name.endswith(".op_stats.pb")
            and "op_stats" not in f.name
            and "hlo_proto" not in f.name
            and not f.name.startswith(".")
        ] or [
            f
            for f in (
                sorted(p.glob("**/*.json.gz")) + sorted(p.glob("**/*.json"))
            )
            if not f.name.startswith(".")
        ]
    except (OSError, ValueError):
      files = []

  if not files:
    return "NO_TRACE_INPUTS"

  hasher = hashlib.sha256()
  for f in files:
    try:
      if f.is_file():
        st = f.stat()
        hasher.update(f"{f.name}:{st.st_size}:".encode("utf-8"))
        with open(f, "rb") as fp:
          hasher.update(fp.read(64 * 1024))
          if st.st_size > 128 * 1024:
            fp.seek(-64 * 1024, 2)
            hasher.update(fp.read(64 * 1024))
    except (OSError, ValueError):
      continue

  return hasher.hexdigest()[:16]
