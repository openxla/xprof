"""Tool to create and query an events DB."""

from collections.abc import Sequence
import hashlib
import json
import pathlib
import tempfile
import textwrap
from typing import Any

import duckdb

from xprof.convert.events_db.python import pywrap_events_db_c_api as events_db
from xprof import version
from xprof.cli.internal import decorators
from xprof.cli.internal.oss import xprof_client

_CACHE_SUBDIR = "events_db"
_DB_FILENAME = "events.parquet"
_KEY_PATH = "path"
_KEY_PATHS = "paths"
_KEY_SKIPPED = "skipped"
_MAX_TRACE_FILES = 64
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


def _multi_cache_dir(db_paths: Sequence[pathlib.Path]) -> pathlib.Path:
  """Returns the shared cache directory for a multi-file Events DB set."""
  combo_digest = _digest(
      version.__version__,
      _VERSION,
      *(str(p.resolve()) for p in db_paths),
  )
  return decorators.get_cache_dir() / _CACHE_SUBDIR / f"multi_{combo_digest}"


def _cache_path_for_query(
    db_paths: Sequence[pathlib.Path] | pathlib.Path, query: str
) -> pathlib.Path:
  """Returns the cache path for a SQL query result across one or more DB paths."""
  if isinstance(db_paths, pathlib.Path):
    return db_paths / f"{_digest(query)}.json"
  if not db_paths:
    raise ValueError("No Events DB Parquet files resolved.")
  if len(db_paths) == 1:
    return db_paths[0].parent / f"{_digest(query)}.json"
  return _multi_cache_dir(db_paths) / f"{_digest(query)}.json"


def _resolve_input_paths(
    client: xprof_client.LocalXprofClient,
    source: str | None,
) -> tuple[pathlib.Path, ...]:
  """Resolves all input trace file paths for the events DB."""
  if source and (source_path := pathlib.Path(source).expanduser()).is_file():
    search_target = source_path
  else:
    search_target = client.get_run_dir(source)

  raw_paths = client.get_xspace_paths(search_target)
  return tuple(pathlib.Path(p) for p in raw_paths)


def _to_json_str(
    path: pathlib.Path,
    skipped: bool,
    paths: Sequence[pathlib.Path] | None = None,
) -> str:
  payload: dict[str, Any] = {_KEY_PATH: str(path), _KEY_SKIPPED: skipped}
  if paths is not None and len(paths) > 1:
    payload[_KEY_PATHS] = [str(p) for p in paths]
  return json.dumps(payload)


def _create_single_events_db(
    source_path: pathlib.Path,
    *,
    bypass_cache: bool = False,
) -> tuple[pathlib.Path, bool]:
  """Converts a single XSpace file to an Events DB Parquet file."""
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


def _combine_events_db_parquets(
    db_paths: Sequence[pathlib.Path],
    *,
    bypass_cache: bool = False,
) -> tuple[pathlib.Path, bool]:
  """Combines multiple per-host Events DB Parquet files into one Parquet file."""
  if not db_paths:
    raise ValueError("No Events DB Parquet files resolved.")
  target_path = _multi_cache_dir(db_paths) / _DB_FILENAME
  if target_path.exists() and not bypass_cache:
    return target_path, True

  target_path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      dir=target_path.parent,
      prefix=f".{target_path.name}.",
      suffix=".tmp",
      delete=False,
  ) as tmp_file:
    tmp_path = pathlib.Path(tmp_file.name)

  escaped_db_paths = [str(p.resolve()).replace("'", "''") for p in db_paths]
  tmp_path_escaped = str(tmp_path.resolve()).replace("'", "''")
  joined = ", ".join(f"'{p}'" for p in escaped_db_paths)
  try:
    with duckdb.connect(":memory:") as con:
      con.execute(
          f"COPY (SELECT * FROM read_parquet([{joined}])) "
          f"TO '{tmp_path_escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)"
      )
    tmp_path.replace(target_path)
  finally:
    tmp_path.unlink(missing_ok=True)

  return target_path, False


def _create_events_db_multi_impl(
    session_id: str | None = None,
    *,
    bypass_cache: bool = False,
) -> tuple[tuple[pathlib.Path, ...], bool]:
  """Creates Events DB Parquet files for all resolved trace files."""
  client = xprof_client.get_client()
  source_paths = _resolve_input_paths(client, session_id)
  if not source_paths:
    raise ValueError(
        f"No .xplane.pb trace files found for session_id={session_id!r}."
    )
  if len(source_paths) > _MAX_TRACE_FILES:
    raise ValueError(
        f"Too many trace files ({len(source_paths)} > {_MAX_TRACE_FILES}); "
        "specify a single .xplane.pb file path."
    )
  db_paths: list[pathlib.Path] = []
  all_skipped = True
  for source_path in source_paths:
    target_path, skipped = _create_single_events_db(
        source_path, bypass_cache=bypass_cache
    )
    db_paths.append(target_path)
    all_skipped = all_skipped and skipped
  return tuple(db_paths), all_skipped


def create_events_db(
    session_id: str | None = None,
    *,
    bypass_cache: bool = False,
) -> str:
  """Converts an XProf XSpace (or multi-host directory) to Events DB Parquet(s).

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
      - path: Path to the generated or existing Parquet file (a combined Parquet
        file across all trace files when multiple `.xplane.pb` files are
        present).
      - paths: List of all per-file Parquet files when multiple trace files are
        present.
      - skipped: True if creation was skipped because all files already exist.
  """
  db_paths, skipped = _create_events_db_multi_impl(
      session_id, bypass_cache=bypass_cache
  )
  if not db_paths:
    raise ValueError("No Events DB Parquet files resolved.")
  if len(db_paths) == 1:
    return _to_json_str(db_paths[0], skipped)
  combined_path, combined_skipped = _combine_events_db_parquets(
      db_paths, bypass_cache=bypass_cache or not skipped
  )
  return _to_json_str(
      combined_path, skipped and combined_skipped, paths=db_paths
  )


def _extract_select_query(query: str | None) -> str:
  """Validates that `query` is a single SELECT statement and returns clean SQL."""
  if not query or not query.strip():
    raise ValueError("SQL query must not be empty.")
  try:
    statements = duckdb.extract_statements(query.strip())
  except duckdb.Error as e:
    raise ValueError(f"Invalid SQL query: {e!r}") from e

  # Queries containing only comments (e.g. "-- comment") or semicolons parse
  # without error but produce an empty statement list.
  if not statements:
    raise ValueError("SQL query must not be empty.")
  if len(statements) > 1:
    raise ValueError(
        f"Expected a single SQL statement, got {len(statements)} statements."
    )
  stmt = statements[0]
  if stmt.type != duckdb.StatementType.SELECT:
    raise ValueError(
        f"Only SELECT queries are allowed, got {stmt.type.name!r}."
    )
  return stmt.query


def _is_select_query(text: str) -> bool:
  """Returns True if `text` parses as a valid DuckDB SELECT statement."""
  try:
    _extract_select_query(text)
    return True
  except ValueError:
    return False


def _query_events_db_impl(
    session_id: str | None = None,
    query: str | None = None,
    *,
    bypass_cache: bool = False,
) -> tuple[pathlib.Path, bool]:
  """Implementation of query_events_db."""
  if (
      (not query or not query.strip())
      and session_id
      and _is_select_query(session_id)
  ):
    query, session_id = session_id, None

  clean_query = _extract_select_query(query)
  db_paths, _ = _create_events_db_multi_impl(session_id, bypass_cache=False)
  if not db_paths:
    raise ValueError("No Events DB Parquet files resolved.")

  target_path = _cache_path_for_query(db_paths, clean_query)
  if target_path.exists() and not bypass_cache:
    return target_path, True

  target_path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      dir=target_path.parent,
      prefix=f".{target_path.name}.",
      suffix=".tmp",
      delete=False,
  ) as tmp_file:
    tmp_path = pathlib.Path(tmp_file.name)

  escaped_db_paths = [str(p.resolve()).replace("'", "''") for p in db_paths]
  tmp_path_escaped = str(tmp_path.resolve()).replace("'", "''")
  if len(escaped_db_paths) == 1:
    parquet_source = f"'{escaped_db_paths[0]}'"
  else:
    joined = ", ".join(f"'{p}'" for p in escaped_db_paths)
    parquet_source = f"[{joined}]"
  allowed_list = ", ".join(
      f"'{p}'" for p in [*escaped_db_paths, tmp_path_escaped]
  )

  try:
    with duckdb.connect(":memory:") as con:
      con.execute(
          f"CREATE VIEW Events AS SELECT * FROM read_parquet({parquet_source})"
      )
      con.execute(textwrap.dedent(f"""\
          SET allowed_paths = [{allowed_list}];
          SET enable_external_access = false;
          SET lock_configuration = true;"""))
      # Define a temporary view first rather than interpolating `clean_query`
      # inside `COPY (...)`. `stmt.query` from `extract_statements()` retains
      # trailing semicolons and single-line `--` comments: inside `COPY (...)`,
      # a semicolon raises a syntax error (`COPY (SELECT ...;)`) and a trailing
      # `--` comment swallows the closing `) TO ...` clause. At the end of
      # `CREATE TEMP VIEW ... AS`, both are valid top-level SQL, and the
      # subsequent `COPY` statement remains completely static.
      view_name = f"QueryResult_{target_path.stem}"
      con.execute(f"CREATE TEMP VIEW {view_name} AS {clean_query}")
      con.execute(
          f"COPY (SELECT * FROM {view_name}) TO '{tmp_path_escaped}' "
          "(FORMAT JSON, ARRAY true, USE_TMP_FILE false)"
      )
    tmp_path.replace(target_path)
  finally:
    tmp_path.unlink(missing_ok=True)

  return target_path, False


def query_events_db_rows(
    session_id: str | None = None,
    query: str | None = None,
    *,
    bypass_cache: bool = False,
) -> list[dict[str, Any]]:
  """Executes a SQL query against an Events DB and returns parsed rows."""
  output_path, _ = _query_events_db_impl(
      session_id, query, bypass_cache=bypass_cache
  )
  return json.loads(output_path.read_text(encoding="utf-8"))


def query_events_db(
    session_id: str | None = None,
    query: str | None = None,
    *,
    bypass_cache: bool = False,
) -> str:
  """Executes a SQL query against an XProf Events DB and saves results to JSON.

  Registers the Events DB Parquet file as a SQL view named `Events` so queries
  can reference `FROM Events` directly. If the Parquet file does not exist yet,
  it is generated automatically. Example query:
  ```sql
  SELECT name, COUNT(*) FROM Events GROUP BY name
  ```

  Query results are cached in the events DB directory. If the query result
  already exists and `bypass_cache` is `False`, it is returned immediately
  with `skipped=True`. Note that `bypass_cache` only applies to the query
  result cache; it does not regenerate the underlying Events DB Parquet file.
  To force regeneration of the Parquet file itself, use `create_events_db` with
  `bypass_cache=True`.

  Args:
    session_id: Session ID, logdir path, run directory, or direct `.xplane.pb` /
      `.xspace.pb` trace file. If omitted and a single SQL SELECT query is
      passed as the first argument, it is treated as `query`.
    query: SQL query to execute against the `Events` view.
    bypass_cache: Whether to bypass the query result cache and re-execute the
      SQL query. Note that this does not regenerate the underlying Parquet
      database (use `create_events_db` with `bypass_cache=True` for that).

  Returns:
    A JSON-formatted string containing:
      - path: Path to the generated or existing JSON query result file.
      - skipped: True if query execution was skipped because the cached result
        already exists.
  """
  output_path, skipped = _query_events_db_impl(
      session_id, query, bypass_cache=bypass_cache
  )
  return _to_json_str(output_path, skipped)
