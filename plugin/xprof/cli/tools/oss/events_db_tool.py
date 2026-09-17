"""Tool to create and query an events DB."""

from collections.abc import Sequence
import hashlib
import json
import pathlib
import pprint
import tempfile


from xprof.convert.events_db.python import pywrap_events_db_c_api as events_db
from xprof import version
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client

_CACHE_SUBDIR = "events_db"
_DB_FILENAME = "events.parquet"
_KEY_PATH = "path"
_KEY_SKIPPED = "skipped"
_MAX_DISPLAY_FILES = 10
_VERSION = "1"


def _digest(*parts: str) -> str:
  """Returns a stable hash of `parts` for use in cache file or folder names."""
  hasher = hashlib.sha256()
  for part in parts:
    encoded = part.encode("utf-8")
    hasher.update(len(encoded).to_bytes(4, "big"))
    hasher.update(encoded)
  return hasher.hexdigest()


def _cache_path_for_create(source_path: pathlib.Path) -> pathlib.Path:
  path = source_path.expanduser().resolve()
  digest = _digest(
      version.__version__,
      _VERSION,
      str(path),
      str(path.stat().st_mtime_ns),
      str(path.stat().st_size),
  )
  return decorators.get_cache_dir() / _CACHE_SUBDIR / digest / _DB_FILENAME


def _create_multi_file_error(
    search_target: pathlib.Path, all_files: Sequence[str]
) -> NotImplementedError:
  displayed_files = pprint.pformat(all_files[:_MAX_DISPLAY_FILES], indent=2)
  truncation_msg = (
      f"\n... and {len(all_files) - _MAX_DISPLAY_FILES} more."
      if len(all_files) > _MAX_DISPLAY_FILES
      else ""
  )
  return NotImplementedError(
      f"Multiple ({len(all_files)}) trace files found in {search_target}:\n"
      f"{displayed_files}{truncation_msg}\n"
      "Multi-file Events DB generation is not supported yet. Please pass a "
      "single .xplane.pb or .xspace.pb file path directly."
  )


def _resolve_input_path(
    client: xprof_client.LocalXprofClient,
    source: str | None,
) -> pathlib.Path:
  """Resolves the input trace file path for the events DB."""
  if source and (source_path := pathlib.Path(source).expanduser()).is_file():
    search_target = source_path
  else:
    search_target = client.get_run_dir(source)

  input_path, *must_be_empty = client.get_xspace_paths(search_target)
  if must_be_empty:
    raise _create_multi_file_error(search_target, (input_path, *must_be_empty))
  return pathlib.Path(input_path)


def _to_json_str(path: pathlib.Path, skipped: bool) -> str:
  return json.dumps({_KEY_PATH: str(path), _KEY_SKIPPED: skipped})


def _create_events_db_impl(
    session_id: str | None = None,
    *,
    bypass_cache: bool = False,
) -> tuple[pathlib.Path, bool]:
  """Implementation of create_events_db."""
  client = xprof_client.get_client()
  source_path = _resolve_input_path(client, session_id)
  target_path = _cache_path_for_create(source_path)
  if target_path.exists() and not bypass_cache:
    return (target_path, True)

  target_path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      dir=target_path.parent,
      prefix=f".{target_path.name}.",
      suffix=".tmp",
      delete=False,
  ) as tmp_file:
    tmp_path = pathlib.Path(tmp_file.name)

  try:
    events_db.xspace_to_parquet(
        input_path=source_path,
        output_path=tmp_path,
        options=events_db.ParquetExportOptions(
            compression_type=events_db.ArrowCompressionType.ZSTD,
            compression_level=3,
        ),
    )
    tmp_path.replace(target_path)
  finally:
    tmp_path.unlink(missing_ok=True)

  return (target_path, False)


def create_events_db(
    session_id: str | None = None,
    *,
    bypass_cache: bool = False,
) -> str:
  """Converts an XProf XSpace to an Events DB Parquet file.

  Destination path is determined using the input session ID. If it already
  exists and `bypass_cache` is `False`, it is returned immediately with
  `skipped=True`.

  Args:
    session_id: Session ID, logdir path, run directory, or direct `.xplane.pb` /
      `.xspace.pb` trace file.
    bypass_cache: Whether to bypass cache and overwrite an existing Parquet
      file.

  Returns:
    A JSON-formatted string containing:
      - path: Path to the generated or existing Parquet file.
      - skipped: True if creation was skipped because the file already exists.
  """
  output_path, skipped = _create_events_db_impl(
      session_id, bypass_cache=bypass_cache
  )
  return _to_json_str(output_path, skipped)
