"""Tool to import a raw trace file into a local logdir for analysis."""

from collections.abc import Sequence
import json
import logging
import pathlib
import shutil

from xprof.cli.internal.oss import xprof_client


VALID_TRACE_EXTENSIONS: tuple[str, ...] = ('.xplane.pb', '.xspace.pb')


def upload_trace(
    file_path: str,
    ttl: int | None = None,
    tag: Sequence[str] = (),
    run_name: str | None = None,
    **kwargs,
) -> str:
  """Imports a trace file into the local logdir under the specified run_name.

  Args:
    file_path: Path to the source trace file (.xplane.pb or .xspace.pb).
    ttl: Time-to-live in seconds (ignored in local mode).
    tag: Tags for the trace (ignored in local mode).
    run_name: Name of the run (session) to import into.
    **kwargs: Additional parameters (ignored).

  Returns:
    JSON status string.

  Raises:
    ValueError: If logdir is unset, file extension is invalid, or run_name
      attempts directory traversal.
    FileNotFoundError: If the source trace file does not exist.
    OSError: If the source trace file cannot be read.
    RuntimeError: If copying the trace to the destination logdir fails.
  """
  del ttl, tag, kwargs  # Unused in local mode.
  client = xprof_client.get_client()
  if not client.logdir:
    raise ValueError('Logdir not set in client. Provide logdir.')

  src_path = pathlib.Path(file_path)
  if not src_path.exists():
    raise FileNotFoundError(f"Source trace file '{file_path}' does not exist.")

  if not any(src_path.name.endswith(ext) for ext in VALID_TRACE_EXTENSIONS):
    raise ValueError(
        f"Unsupported file format '{src_path.name}'. Only XSpace formats"
        f" ({', '.join(VALID_TRACE_EXTENSIONS)}) are supported by XProf."
    )

  run_name = run_name or 'imported_trace'
  if (
      '/' in run_name
      or '\\' in run_name
      or '..' in run_name
      or pathlib.PurePath(run_name).is_absolute()
  ):
    raise ValueError(
        f"Invalid run_name '{run_name}': must be a single directory name"
        ' without path separators or traversal components.'
    )

  base_dir = (client.logdir / 'plugins' / 'profile').resolve()
  dest_dir = (base_dir / run_name).resolve()
  if not dest_dir.is_relative_to(base_dir) or dest_dir == base_dir:
    raise ValueError(
        f"Invalid run_name '{run_name}': resolves outside the logdir profile"
        ' hierarchy.'
    )

  try:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src_path.name
    logging.info('Copying trace from %s to %s', src_path, dest_path)
    shutil.copy(src_path, dest_path)

    return json.dumps(
        {
            'status': 'success',
            'message': f"Successfully imported trace to run '{run_name}'",
            'run_name': run_name,
            'run_path': str(dest_dir),
            'imported_file': str(dest_path),
        },
        indent=2,
    )

  except (OSError, ValueError):
    raise
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception('Failed to import trace %s to %s', file_path, dest_dir)
    raise RuntimeError(f'Failed to import trace: {e}') from e
