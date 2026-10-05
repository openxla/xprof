"""Decorators for caching."""

import atexit
import collections
from collections.abc import Callable, Collection, Sequence
import contextlib
import copy
import functools
import getpass
import hashlib
import json
import pathlib
import random
import sqlite3
import sys
import tempfile
import textwrap
import threading
import time
from typing import Any, TypeVar

from absl import logging

_T = TypeVar("_T")

_UNKNOWN = object()

_CUSTOM_CALL_TUNE_DOC: str = (
    "https://openxla.org/xprof/custom_call_profiling#how-to-tune"
)

LLO_DATA_ABSENT_REMEDIATION: str = (
    "To enable LLO tracing, ensure the workload is executed with"
    ' LIBTPU_INIT_ARGS="--xla_xprof_register_llo_debug_info=true"'
    " exported strictly BEFORE 'import jax'. Adding"
    " --xla_xprof_enable_custom_call_tracing=true captures fine-grained"
    " runtime LLO details and increases trace size; if trace buffer overflow"
    " drops events, tune the vtrace frequency flag"
    " (trace_best_effort_frequency / trace_guaranteed_frequency in"
    f" xla_tpu_bundle_instrumentation_options; see {_CUSTOM_CALL_TUNE_DOC})."
    " Prerequisites: Python 3.11+ (Python 3.12 recommended via uv),"
    " JAX >= 0.11.0 (default Cloud TPU VM images running Python 3.10 cap JAX"
    " at 0.6.2 and lack LLO flag support), and xprof-nightly."
)

_FINGERPRINT_CACHE_MAXSIZE = 1024
_FINGERPRINT_MEM_CACHE: collections.OrderedDict[Any, str] = (
    collections.OrderedDict()
)


class Cache:
  """A minimal, persistent, SQLite-backed cache.

  Attributes:
    directory: The directory where the database file is stored.
    db_path: The full path to the SQLite database file.
  """

  UNKNOWN = _UNKNOWN

  def __init__(self, directory: pathlib.Path, **kwargs):
    """Initializes the instance.

    Args:
      directory: The directory where the database file will be stored.
      **kwargs: Unused parameters absorbed for compatibility.
    """
    self._size_limit = kwargs.get("size_limit")
    self.directory = directory
    self.db_path = directory / "cache.db"
    self._lock = threading.RLock()
    self._mem_cache: dict[str, tuple[Any, float | None, float]] = {}
    self._conn: sqlite3.Connection | None = None
    self._init_db()

  def _init_db(self):
    """Initializes the SQLite database and table, pruning expired entries."""
    with self._lock:
      self._mem_cache.clear()
      _FINGERPRINT_MEM_CACHE.clear()
      if self._conn is not None:
        try:
          self._conn.close()
        except sqlite3.Error:
          pass
        self._conn = None
      self.directory.mkdir(parents=True, exist_ok=True)
      with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(textwrap.dedent("""
          CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT,
            expire REAL,
            set_time REAL
          )
        """))
        # Prune expired entries on startup.
        conn.execute(
            "DELETE FROM cache WHERE expire IS NOT NULL AND expire < ?",
            (time.time(),),
        )
        conn.commit()

  def _get_conn(self):
    """Returns a SQLite connection configured with WAL mode."""
    conn = sqlite3.connect(self.db_path)
    try:
      conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
      pass
    return conn

  def _get_persistent_conn(self) -> sqlite3.Connection:
    """Returns a reusable persistent connection for fast reads/writes."""
    with self._lock:
      if self._conn is None or not self.db_path.exists():
        if not self.db_path.exists():
          self._init_db()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
          self._conn.execute("PRAGMA journal_mode=WAL")
          self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
          pass
      return self._conn

  def flush(self) -> None:
    """No-op retained for backwards compatibility (writes are immediate)."""
    return

  def get(self, key: str, default: Any = _UNKNOWN) -> Any:
    """Retrieves a value from the cache."""
    res, _ = self.get_with_metadata(key, default=default)
    return res

  def get_with_metadata(
      self, key: str, default: Any = _UNKNOWN
  ) -> tuple[Any, float | None]:
    """Retrieves value and set_time metadata from the cache."""
    with self._lock:
      mem_entry = self._mem_cache.get(key)
      if mem_entry is not None:
        val, expire, set_time = mem_entry
        if expire is not None and expire < time.time():
          self._mem_cache.pop(key, None)
        elif self.db_path.exists():
          return copy.deepcopy(val), set_time
        else:
          self._mem_cache.clear()

      try:
        conn = self._get_persistent_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value, expire, set_time FROM cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
          return default, None
        value_str, expire, set_time = row
        if expire is not None and expire < time.time():
          self.delete(key)
          return default, None
        if value_str is None:
          return default, None
        try:
          parsed = json.loads(value_str)
          self._mem_cache[key] = (parsed, expire, set_time)
          return copy.deepcopy(parsed), set_time
        except json.JSONDecodeError:
          self.delete(key)
          return default, None
      except sqlite3.Error:
        return default, None

  def set(self, key: str, value: Any, expire: float | None = None, **kwargs):
    """Stores a value in the cache."""
    del kwargs  # Unused absorbed for compatibility.
    if _is_error_payload(value):
      return

    # This will raise a TypeError if the value is bytes (not JSON serializable).
    val_str = json.dumps(value)
    canonical_val = json.loads(val_str)
    now = time.time()
    expire_time = now + expire if expire is not None else None
    with self._lock:
      self._mem_cache[key] = (canonical_val, expire_time, now)
      try:
        conn = self._get_persistent_conn()
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expire, set_time)"
            " VALUES (?, ?, ?, ?)",
            (key, val_str, expire_time, now),
        )
        conn.commit()
      except sqlite3.Error:
        pass

  def delete(self, key: str):
    """Deletes a key from the cache."""
    with self._lock:
      self._mem_cache.pop(key, None)
      try:
        conn = self._get_persistent_conn()
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
      except sqlite3.Error:
        pass

  @contextlib.contextmanager
  def transact(self):
    """Acquires a transaction lock on the database."""
    with self._lock:
      conn = self._get_conn()
      try:
        conn.execute("BEGIN IMMEDIATE")
        try:
          yield
          conn.commit()
        except Exception:
          if conn.in_transaction:
            conn.rollback()
          raise
      finally:
        conn.close()

  def close(self) -> None:
    """Closes the persistent cache connection."""
    with self._lock:
      if self._conn is not None:
        try:
          self._conn.close()
        except sqlite3.Error:
          pass
        self._conn = None


def _is_error_payload(value: Any) -> bool:
  """Checks if a payload represents an error that should not be cached."""
  if isinstance(value, dict):
    return "error" in value or value.get("status") in (
        "ERROR",
        "INTERNAL_ERROR",
    )
  if isinstance(value, str):
    val_strip = value.strip()
    if val_strip.startswith("{") and val_strip.endswith("}"):
      try:
        data = json.loads(val_strip)
        if isinstance(data, dict):
          return "error" in data or data.get("status") in (
              "ERROR",
              "INTERNAL_ERROR",
          )
      except (json.JSONDecodeError, TypeError):
        pass
  return False


@functools.lru_cache(maxsize=1)
def _get_xprof_version() -> str:
  """Retrieves the xprof version string for cache key versioning."""
  try:
    import importlib.metadata  # pylint: disable=g-import-not-at-top

    return importlib.metadata.version("xprof")
  except Exception:  # pylint: disable=broad-except
    pass
  try:
    for mod_name in (
        "google3.third_party.xprof.plugin.xprof.version",
        "xprof.version",
    ):
      if mod_name in sys.modules:
        mod = sys.modules[mod_name]
        return getattr(mod, "__version__", "unknown")
    import importlib  # pylint: disable=g-import-not-at-top

    for mod_name in (
        "google3.third_party.xprof.plugin.xprof.version",
        "xprof.version",
    ):
      try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "unknown")
      except Exception:  # pylint: disable=broad-except
        pass
  except Exception:  # pylint: disable=broad-except
    pass
  return "unknown"


def _resolve_session_path_via_client(val: str) -> pathlib.Path | None:
  """Dynamically resolves a session ID against the active xprof client if present."""
  mod_names = ("xprof.cli.internal.oss.xprof_client",)
  for mod_name in mod_names:
    client = None
    try:
      if mod_name in sys.modules:
        mod = sys.modules[mod_name]
      else:
        import importlib  # pylint: disable=g-import-not-at-top

        mod = importlib.import_module(mod_name)
      get_client_fn = getattr(mod, "get_client", None)
      if callable(get_client_fn):
        client = get_client_fn()
    except Exception:  # pylint: disable=broad-except
      client = None

    if (
        client
        and getattr(client, "_logdir", None)
        and hasattr(client, "get_run_dir")
    ):
      try:
        resolved = client.get_run_dir(val)
        if resolved and pathlib.Path(resolved).exists():
          return pathlib.Path(resolved)
      except Exception:  # pylint: disable=broad-except
        pass
  return None


def _safe_path_exists(val: str) -> bool:
  """Safely checks whether a string refers to an existing local path."""
  if not val or len(val) > 1024 or "\n" in val or "\x00" in val:
    return False
  try:
    return pathlib.Path(val).expanduser().exists()
  except (OSError, RuntimeError, ValueError):
    return False


def _compute_path_fingerprint(
    val: Any, xspace_paths: Sequence[str] | None = None
) -> str:
  """Computes a content-addressable SHA-256 fingerprint covering raw trace inputs."""
  if not isinstance(val, (str, pathlib.Path)) and not xspace_paths:
    return "NO_TRACE_INPUTS"

  if (
      not xspace_paths
      and isinstance(val, str)
      and bool(val)
      and not _safe_path_exists(val)
  ):
    if len(val) <= 1024 and "\n" not in val and "\x00" not in val:
      resolved_p = _resolve_session_path_via_client(val)
      if resolved_p is not None:
        _FINGERPRINT_MEM_CACHE.pop(val, None)
        return _compute_path_fingerprint_uncached(resolved_p, xspace_paths)
    cached_fp = _FINGERPRINT_MEM_CACHE.get(val)
    if cached_fp is not None:
      _FINGERPRINT_MEM_CACHE.move_to_end(val)
      return cached_fp

    res = _compute_path_fingerprint_uncached(val, xspace_paths)
    if res in ("NONEXISTENT", "NO_TRACE_INPUTS"):
      _FINGERPRINT_MEM_CACHE[val] = res
      _FINGERPRINT_MEM_CACHE.move_to_end(val)
      while len(_FINGERPRINT_MEM_CACHE) > _FINGERPRINT_CACHE_MAXSIZE:
        _FINGERPRINT_MEM_CACHE.popitem(last=False)
    return res

  return _compute_path_fingerprint_uncached(val, xspace_paths)


def _compute_path_fingerprint_uncached(
    val: Any, xspace_paths: Sequence[str] | None = None
) -> str:
  """Computes a fingerprint string for a session path or xspace file list."""
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
    if isinstance(val, str) and (
        len(val) > 1024 or "\n" in val or "\x00" in val
    ):
      return "NONEXISTENT"
    try:
      p = pathlib.Path(val).expanduser()
      if not p.exists():
        resolved_p = _resolve_session_path_via_client(str(val))
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
    except (OSError, RuntimeError, ValueError):
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
            for f in sorted(p.glob("**/*.json.gz")) + sorted(
                p.glob("**/*.json")
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


def _add_cache_indicator(value: Any, set_time: float | None = None) -> Any:
  """Adds cache indicators to the value if it's a dict or JSON string."""
  if isinstance(value, dict):
    res = {**value, "__cached__": True}
    if set_time is not None:
      res["__cache_age_s__"] = round(time.time() - set_time, 2)
    return res
  if isinstance(value, str):
    try:
      data = json.loads(value)
      if isinstance(data, dict):
        data["__cached__"] = True
        if set_time is not None:
          data["__cache_age_s__"] = round(time.time() - set_time, 2)
        return json.dumps(data)
    except json.JSONDecodeError:
      pass
  return value


def _get_cache_dir() -> pathlib.Path:
  """Returns a user-specific temporary directory for the cache."""
  user = getpass.getuser()
  cache_dir = pathlib.Path(tempfile.gettempdir()) / f"xprof_cli_cache_{user}"
  cache_dir.mkdir(mode=0o700, exist_ok=True)
  return cache_dir


get_cache_dir = _get_cache_dir
compute_path_fingerprint = _compute_path_fingerprint


_GLOBAL_CACHE: Cache | None = None


def get_cache() -> Cache:
  """Returns the global Cache instance, initializing it lazily."""
  global _GLOBAL_CACHE
  if _GLOBAL_CACHE is None:
    # We use a size limit of 1GB and a default expiration of 1 hour.
    # This is a global resource that lives for the lifetime of the CLI process.
    # We register an atexit handler to ensure the underlying database connection
    # is closed.
    _GLOBAL_CACHE = Cache(
        _get_cache_dir(),
        size_limit=1024 * 1024 * 1024,
    )
    atexit.register(_GLOBAL_CACHE.close)
  return _GLOBAL_CACHE


def _extract_bypass_cache(
    args: tuple[Any, ...],
    kwargs_call: dict[str, Any],
    bypass_cache_idx: int | None,
) -> tuple[bool, tuple[Any, ...]]:
  """Extracts bypass_cache and returns (bypass_cache, args_for_cache_key)."""
  if bypass_cache_idx is None:
    return bool(kwargs_call.pop("bypass_cache", False)), args
  if "bypass_cache" in kwargs_call:
    return bool(kwargs_call["bypass_cache"]), args
  if len(args) > bypass_cache_idx:
    return bool(args[bypass_cache_idx]), (
        args[:bypass_cache_idx] + args[bypass_cache_idx + 1 :]
    )
  return False, args


def cached(
    *,
    cache: Cache | None = None,
    expire: float | None = 3600,
    ignore: Collection[str] = (),
    **kwargs,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
  """Caches the result of a function call to disk.

  Args:
    cache: Optional cache instance. If not provided, uses the global cache.
    expire: Time in seconds before the cache entry expires. Defaults to 1 hour.
    ignore: Tuple of kwarg names to ignore for the cache key.
    **kwargs: Additional arguments passed to Cache.set.

  Returns:
    The decorated function.
  """

  def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
    try:
      import inspect  # pylint: disable=g-import-not-at-top

      func_sig = inspect.signature(func)
      params = list(func_sig.parameters.keys())
      bypass_cache_idx = (
          params.index("bypass_cache") if "bypass_cache" in params else None
      )
    except Exception:  # pylint: disable=broad-except
      func_sig = None
      bypass_cache_idx = None

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs_call: Any) -> _T:
      bypass_cache, args_for_key = _extract_bypass_cache(
          args, kwargs_call, bypass_cache_idx
      )

      # 1. Compute a stable key with path and content signature normalization.
      key_kwargs = {
          k: v
          for k, v in kwargs_call.items()
          if k not in ignore and k != "bypass_cache"
      }

      normalized_args = []
      fingerprints = []
      for arg in args_for_key:
        fp = _compute_path_fingerprint(arg)
        if fp:
          fingerprints.append(fp)
        if fp not in ("NO_TRACE_INPUTS", "NONEXISTENT"):
          normalized_args.append(f"trace_sig:{fp}")
        else:
          normalized_args.append(arg)

      normalized_kwargs = {}
      for k, v in key_kwargs.items():
        fp = _compute_path_fingerprint(v)
        if fp:
          fingerprints.append(fp)
        if fp not in ("NO_TRACE_INPUTS", "NONEXISTENT"):
          normalized_kwargs[k] = f"trace_sig:{fp}"
        else:
          normalized_kwargs[k] = v

      fingerprint_str = ";".join(fingerprints)
      try:
        # Sort items to ensure order stability for JSON dict kwargs.
        key_kwargs_sorted = sorted(normalized_kwargs.items())
        key = json.dumps(
            [
                getattr(func, "__module__", ""),
                getattr(func, "__qualname__", ""),
                _get_xprof_version(),
                normalized_args,
                key_kwargs_sorted,
                fingerprint_str,
            ],
            sort_keys=True,
        )
      except Exception:  # pylint: disable=broad-except
        # Caching is a best-effort optimization. If we fail to serialize the
        # arguments to create a cache key (e.g. non-serializable objects),
        # it is safe to just execute the function directly.
        logging.warning(
            "Failed to create cache key, calling function directly",
            exc_info=True,
        )
        return func(*args, **kwargs_call)

      cache_instance = cache if cache is not None else get_cache()
      if not bypass_cache:
        # 2. Check the cache.
        value = _UNKNOWN
        set_time = None
        if hasattr(cache_instance, "get_with_metadata"):
          try:
            res = cache_instance.get_with_metadata(key, default=_UNKNOWN)
            if (
                isinstance(res, tuple)
                and len(res) == 2
                and "Mock" not in type(res).__name__
            ):
              value, set_time = res
            else:
              value = cache_instance.get(key, default=_UNKNOWN)
          except Exception:  # pylint: disable=broad-except
            value = cache_instance.get(key, default=_UNKNOWN)
        else:
          value = cache_instance.get(key, default=_UNKNOWN)

        if value is not _UNKNOWN:
          logging.debug("Cache hit for %s", getattr(func, "__name__", ""))
          return _add_cache_indicator(value, set_time=set_time)

      # 3. MISS or BYPASS.
      logging.debug("Cache miss for %s", getattr(func, "__name__", ""))
      result = func(*args, **kwargs_call)

      # 4. Store in cache.
      try:
        cache_instance.set(key, result, expire=expire, **kwargs)
      except Exception:  # pylint: disable=broad-except
        logging.warning("Failed to store in cache", exc_info=True)

      return result

    # Add bypass_cache to the signature if not present.
    if func_sig is not None and bypass_cache_idx is None:
      try:
        import inspect  # pylint: disable=g-import-not-at-top

        params = list(func_sig.parameters.values())
        new_param = inspect.Parameter(
            "bypass_cache",
            inspect.Parameter.KEYWORD_ONLY,
            default=False,
            annotation=bool,
        )
        params.append(new_param)
        new_sig = func_sig.replace(parameters=params)
        wrapper.__signature__ = new_sig  # pyrefly: ignore[missing-attribute]
      except Exception:  # pylint: disable=broad-except
        pass

    return wrapper

  return decorator


class _SharedRateLimiter:
  """Process-safe token bucket rate limiter backed by Cache."""

  def __init__(self, cache_instance: Cache, key: str, rate: float, burst: int):
    self._cache = cache_instance
    self._key = f"ratelimit:{key}"
    self._rate = rate
    self._burst = burst

  def sleep_and_advance(self) -> None:
    """Blocks until a token is available."""
    while True:
      try:
        with self._cache.transact():
          now = time.time()
          tokens, last_update = self._cache.get(
              self._key, default=(self._burst, now)
          )
          elapsed = now - last_update
          tokens = min(self._burst, tokens + elapsed * self._rate)

          if tokens >= 1.0:
            self._cache.set(self._key, (tokens - 1.0, now))
            return

          wait_time = (1.0 - tokens) / self._rate
      except sqlite3.OperationalError:
        # Handle lock contention with a short randomized jitter before retry.
        time.sleep(random.uniform(0.01, 0.05))
        continue
      except (sqlite3.Error, TypeError, ValueError):
        # Fallback best effort if database schema/cache is degraded.
        return

      if wait_time > 5.0:
        logging.info(
            "Rate limit reached for %s. Waiting %.2f seconds before"
            " retrying...",
            self._key,
            wait_time,
        )
      # Add jitter to prevent concurrent thundering herd wakeups.
      time.sleep(max(0.01, wait_time) * random.uniform(0.95, 1.05))


def rate_limited(
    rate: float = 1.0, burst: int = 1
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
  """Rate limits function calls across processes.

  Args:
    rate: The rate limit in calls per second.
    burst: The maximum number of calls that can be made in a burst.

  Returns:
    The decorated function.
  """

  def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
    key = (
        f"{getattr(func, '__module__', '')}.{getattr(func, '__qualname__', '')}"
    )

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
      limiter = _SharedRateLimiter(get_cache(), key, rate, burst)
      limiter.sleep_and_advance()
      return func(*args, **kwargs)

    return wrapper

  return decorator
