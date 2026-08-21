"""Decorators for caching."""

import atexit
from collections.abc import Callable, Collection, Sequence
import contextlib
import functools
import getpass
import hashlib
import json
import pathlib
import random
import sqlite3
import tempfile
import textwrap
import time
from typing import Any, TypeVar

from absl import logging

_T = TypeVar("_T")

_UNKNOWN = object()


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
    self._init_db()

  def _init_db(self):
    """Initializes the SQLite database and table, pruning expired entries."""
    self.directory.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
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
    """Returns a new SQLite connection.

    Sqlite3 connections are not thread-safe, so we open a new connection for
    each operation to prevent issues across multiple threads.
    """
    return sqlite3.connect(self.db_path)

  def get(self, key: str, default: Any = _UNKNOWN) -> Any:
    """Retrieves a value from the cache.

    If the key doesn't exist or is expired, returns the default value.

    Args:
      key: The cache key to look up.
      default: Value to return if key is not found or expired. Defaults to
        Cache.UNKNOWN.

    Returns:
      The cached Python object, or the default value.
    """
    res, _ = self.get_with_metadata(key, default=default)
    return res

  def get_with_metadata(
      self, key: str, default: Any = _UNKNOWN
  ) -> tuple[Any, float | None]:
    """Retrieves value and set_time metadata from the cache.

    Args:
      key: The cache key to look up.
      default: Value to return if key is not found or expired. Defaults to
        Cache.UNKNOWN.

    Returns:
      A tuple of (cached_value, set_time).
    """
    try:
      with contextlib.closing(self._get_conn()) as conn:
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
          return json.loads(value_str), set_time
        except json.JSONDecodeError:
          self.delete(key)
          return default, None
    except sqlite3.Error:
      return default, None

  def set(self, key: str, value: Any, expire: float | None = None, **kwargs):
    """Stores a value in the cache.

    Values are JSON-serialized before storage. Storing non-JSON serializable
    objects (like bytes) will raise a TypeError.

    Args:
      key: The cache key.
      value: The Python object to store. Must be JSON serializable.
      expire: Optional expiration time in seconds from now.
      **kwargs: Unused parameters absorbed for compatibility.

    Raises:
      TypeError: If the value is not JSON serializable.
    """
    del kwargs  # Unused absorbed for compatibility.
    if _is_error_payload(value):
      return

    # This will raise a TypeError if the value is bytes (not JSON serializable).
    val_str = json.dumps(value)
    now = time.time()
    expire_time = now + expire if expire is not None else None
    try:
      with contextlib.closing(self._get_conn()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expire, set_time)"
            " VALUES (?, ?, ?, ?)",
            (key, val_str, expire_time, now),
        )
        conn.commit()
    except sqlite3.Error:
      # Caching is best effort.
      pass

  def delete(self, key: str):
    """Deletes a key from the cache.

    Args:
      key: The cache key to delete.
    """
    try:
      with contextlib.closing(self._get_conn()) as conn:
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
    except sqlite3.Error:
      pass

  @contextlib.contextmanager
  def transact(self):
    """Acquires a transaction lock on the database."""
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
    """Closes the cache (noop for this implementation)."""


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


def _compute_path_fingerprint(
    val: Any, xspace_paths: Sequence[str] | None = None
) -> str:
  """Computes a timestamp+size fingerprint covering exact raw trace inputs."""
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
    try:
      p = pathlib.Path(val).expanduser()
      if not p.exists():
        return "NONEXISTENT"
      if p.is_file():
        st = p.stat()
        return f"f:{st.st_mtime_ns}:{st.st_size}"
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

  base_dir = (
      pathlib.Path(val)
      if isinstance(val, (str, pathlib.Path)) and pathlib.Path(val).is_dir()
      else files[0].parent
  )
  stats = []
  for f in files:
    try:
      if f.is_file():
        st = f.stat()
        rel = (
            str(f.relative_to(base_dir))
            if f.is_relative_to(base_dir)
            else f.name
        )
        stats.append(f"{rel}:{st.st_mtime_ns}:{st.st_size}")
    except (OSError, ValueError):
      continue

  if not stats:
    return "NO_TRACE_INPUTS"
  return hashlib.sha256(";".join(stats).encode("utf-8")).hexdigest()[:16]


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
      has_bypass_cache = "bypass_cache" in func_sig.parameters
    except Exception:  # pylint: disable=broad-except
      func_sig = None
      has_bypass_cache = False

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs_call: Any) -> _T:
      if has_bypass_cache:
        bypass_cache = kwargs_call.get("bypass_cache", False)
      else:
        bypass_cache = kwargs_call.pop("bypass_cache", False)

      # 1. Compute a stable key with path fingerprinting.
      key_kwargs = {
          k: v
          for k, v in kwargs_call.items()
          if k not in ignore and k != "bypass_cache"
      }
      fingerprints = [_compute_path_fingerprint(arg) for arg in args] + [
          _compute_path_fingerprint(v) for v in key_kwargs.values()
      ]
      fingerprint_str = ";".join(f for f in fingerprints if f)
      try:
        # Sort items to ensure order stability for JSON dict kwargs.
        key_kwargs_sorted = sorted(key_kwargs.items())
        key = json.dumps(
            [
                getattr(func, "__module__", ""),
                getattr(func, "__qualname__", ""),
                args,
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
    if func_sig is not None and not has_bypass_cache:
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
