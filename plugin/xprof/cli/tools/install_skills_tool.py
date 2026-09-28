"""Tool to install XProf agent skills into a local workspace directory."""

import argparse
import json
import os
import pathlib
import shutil
from typing import Any


def install_skills(
    *,
    target_dir: str = ".claude/skills/xprof",
    force: bool = False,
    source_dir: str | pathlib.Path | None = None,
) -> str:
  """Non-destructively unpacks bundled XProf agent skills into target_dir.

  Args:
    target_dir: Destination path for the installed skill bundle.
    force: If True, overwrites existing files in target_dir.
    source_dir: Optional override path containing bundled skill files.

  Returns:
    JSON status report of installed files.
  """
  target_path = pathlib.Path(target_dir).resolve()
  target_path.mkdir(parents=True, exist_ok=True)

  resolved_source: pathlib.Path | None = None
  if source_dir is not None:
    candidate = pathlib.Path(source_dir).resolve()
    if candidate.exists() and candidate.is_dir():
      resolved_source = candidate
  else:
    # Locate bundled skills directory relative to this package.
    pkg_root = pathlib.Path(__file__).resolve().parent.parent
    candidate_sources = [
        pkg_root / "skills" / "xprof",
        pkg_root.parent / "skills" / "xprof",
        pkg_root.parent.parent / "skills" / "xprof",
        pkg_root.parent.parent.parent / "skills" / "xprof",
    ]
    for candidate in candidate_sources:
      if candidate.exists() and candidate.is_dir():
        resolved_source = candidate
        break

  if resolved_source is None:
    return json.dumps(
        {
            "status": "ERROR",
            "message": "Bundled skill files not found in package installation.",
            "target_dir": str(target_path),
            "installed_files": [],
            "skipped_files": [],
        },
        indent=2,
    )

  installed_files = []
  skipped_files = []

  for source_file in sorted(resolved_source.rglob("*")):
    if source_file.is_dir():
      continue
    relative_path = source_file.relative_to(resolved_source)
    destination_file = target_path / relative_path
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    if destination_file.exists() and not force:
      skipped_files.append(str(destination_file))
    else:
      if os.path.islink(destination_file):
        os.unlink(destination_file)
      shutil.copy2(source_file, destination_file)
      installed_files.append(str(destination_file))

  result: dict[str, Any] = {
      "status": "SUCCESS",
      "target_dir": str(target_path),
      "installed_files": installed_files,
      "skipped_files": skipped_files,
  }
  return json.dumps(result, indent=2)


def main() -> None:
  """CLI entry point for `xprof-install-skills`."""
  parser = argparse.ArgumentParser(description="Install XProf agent skills")
  parser.add_argument(
      "--target_dir",
      default=".claude/skills/xprof",
      help="Target directory for installed skills",
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Overwrite existing skill files",
  )
  args = parser.parse_args()
  print(install_skills(target_dir=args.target_dir, force=args.force))


if __name__ == "__main__":
  main()
