"""Tool to import a raw trace file into a local logdir for analysis."""

from collections.abc import Sequence
import json
import logging
import pathlib
import shutil

from xprof.cli.internal.oss import xprof_client


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
  """
  del ttl, tag, kwargs  # Unused in local mode.
  client = xprof_client.get_client()
  if not client.logdir:
    return json.dumps({"error": "Logdir not set in client. Provide logdir."})

  run_name = run_name or "imported_trace"
  src_path = pathlib.Path(file_path)
  dest_dir = client.logdir / "plugins" / "profile" / run_name
  try:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src_path.name
    logging.info("Copying trace from %s to %s", src_path, dest_path)
    shutil.copy(src_path, dest_path)

    return json.dumps(
        {
            "status": "success",
            "message": f"Successfully imported trace to run '{run_name}'",
            "run_name": run_name,
            "run_path": str(dest_dir),
            "imported_file": str(dest_path),
        },
        indent=2,
    )

  except FileNotFoundError:
    return json.dumps(
        {"error": f"Source trace file '{file_path}' does not exist."},
        indent=2,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to import trace %s to %s", file_path, dest_dir)
    return json.dumps({"error": f"Failed to import trace: {e!r}"}, indent=2)
