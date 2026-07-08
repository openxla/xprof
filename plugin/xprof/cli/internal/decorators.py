"""Decorators for caching."""

import atexit
from collections.abc import Callable, Collection
import contextlib
import functools
import getpass
import json
import pathlib
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
    try:
      with contextlib.closing(self._get_conn()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value, expire FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
          return default
        value_str, expire = row
        if expire is not None and expire < time.time():
          self.delete(key)
          return default
        if value_str is None:
          return default
        try:
          return json.loads(value_str)
        except json.JSONDecodeError:
          self.delete(key)
          return default
    except sqlite3.Error:
      return default

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
    del kwargs  # Unused by this minimal cache implementation.
    # This will raise a TypeError if the value is bytes (not JSON serializable).
    val_str = json.dumps(value)
    expire_time = time.time() + expire if expire is not None else None
    try:
      with contextlib.closing(self._get_conn()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expire) VALUES (?,"
            " ?, ?)",
            (key, val_str, expire_time),
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

  def close(self) -> None:
    """Closes the cache (noop for this implementation)."""


def _get_cache_dir() -> pathlib.Path:
  """Returns a user-specific temporary directory for the cache."""
  user = getpass.getuser()
  cache_dir = pathlib.Path(tempfile.gettempdir()) / f"xprof_cli_cache_{user}"
  cache_dir.mkdir(mode=0o700, exist_ok=True)
  return cache_dir


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

      # 1. Compute a stable key.
      key_kwargs = {
          k: v
          for k, v in kwargs_call.items()
          if k not in ignore and k != "bypass_cache"
      }
      try:
        # Sort items to ensure order stability for JSON dict kwargs.
        key_kwargs_sorted = sorted(key_kwargs.items())
        key = json.dumps(
            [
                getattr(func, "__module__", ""),
                getattr(func, "__qualname__", ""),
                args,
                key_kwargs_sorted,
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
        value = cache_instance.get(key, default=_UNKNOWN)
        if value is not _UNKNOWN:
          logging.debug("Cache hit for %s", getattr(func, "__name__", ""))
          return value

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
        wrapper.__signature__ = new_sig  # pytype: disable=attribute-error
      except Exception:  # pylint: disable=broad-except
        pass

    return wrapper

  return decorator
