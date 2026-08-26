#!/usr/bin/env python3
"""Overlays this checkout's built frontend onto the served XProf static dir.

Why this exists: `xprof server` serves its UI from the installed distribution's
`xprof/static/` directory. On a normal machine that is a PyPI wheel carrying a
prebuilt bundle, so a UI test suite launched against `xprof` on PATH validates a
*released* frontend and is entirely blind to the code in this checkout. Copying
the locally built assets over that directory is what makes the suite test this
repository.

The asset name mapping mirrors the genrules in frontend/BUILD:
    main.js      -> bundle.js
    polyfills.js -> zone.js
    runtime.js   -> runtime.js
    styles.css   -> styles.css
The repo's own index.html is used too, because it loads runtime.js before
bundle.js whereas the wheel's older loader expects a single self-contained
bundle.

Everything replaced is backed up alongside with a .orig suffix, and --restore
puts it all back. WASM/trace-viewer assets are left untouched: they come from
the C++/Emscripten build, which this script does not attempt to reproduce.

Usage:
    python -m tests.ui.sync_frontend            # overlay
    python -m tests.ui.sync_frontend --restore  # undo
"""

# pylint: disable=g-doc-args,g-doc-return-or-yield,g-short-docstring-punctuation

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys

# runtime.js is deliberately absent: the installed server has no route for it
# and answers with the SPA's index.html, which the browser then refuses to
# execute ("MIME type ('text/html') is not executable"). The webpack runtime is
# instead prepended to bundle.js -- exactly the "runtime is pre-bundled in
# bundle.js" arrangement that plugin/xprof/static/index.html already documents.
ASSET_MAP = {
    "polyfills.js": "zone.js",
    "styles.css": "styles.css",
}
# (runtime.js + main.js) -> bundle.js, concatenated in that order.
CONCAT_BUNDLE = ("runtime.js", "main.js", "bundle.js")
BUNDLE_DIGEST_FILE = ".synced_bundle.sha256"


def repo_root() -> str:
  """Returns repository root directory path."""
  return os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  )


def served_static_dir() -> str:
  """Locates the INSTALLED xprof's static dir.

  A plain `import xprof` is not safe here: run from the repo root, the repo's
  own top-level `xprof/` package (the C++ tree) shadows the installed
  distribution and the import silently resolves to the wrong thing. So the
  repo root is stripped from the search path first.

  Returns:
    Path to the installed static directory.
  """
  root = repo_root()
  clean = [
      p for p in sys.path if p and os.path.realpath(p) != os.path.realpath(root)
  ]
  saved, sys.path[:] = sys.path[:], clean
  try:
    spec = importlib.util.find_spec("xprof")
    if spec is None or not spec.origin:
      sys.exit(
          "xprof is not importable; activate the venv that has it installed."
      )
    pkg_dir = os.path.dirname(os.path.abspath(spec.origin))
  finally:
    sys.path[:] = saved

  if os.path.realpath(pkg_dir) == os.path.realpath(os.path.join(root, "xprof")):
    sys.exit(
        f"Resolved xprof to the repo tree at {pkg_dir}, not an installed "
        "distribution. Install xprof into a venv and re-run."
    )
  static = os.path.join(pkg_dir, "static")
  if not os.path.isdir(static):
    sys.exit(f"Installed xprof has no static dir at {static}")
  return static


def _digest(path: str) -> str:
  with open(path, "rb") as f:
    return hashlib.sha256(f.read()).hexdigest()[:12]


def overlay() -> int:
  """Overlays locally built frontend assets onto installed static.

  directory.
  """
  dist = os.path.join(repo_root(), "frontend", "dist", "xprof")
  static = served_static_dir()
  if not os.path.isdir(dist):
    sys.exit(
        f"No build at {dist}.\nBuild it first:\n"
        "  python3 patch_keys.py\n"
        "  cd frontend && ../node_modules/.bin/ng build"
        " --configuration production"
    )

  for src_name, dst_name in ASSET_MAP.items():
    src = os.path.join(dist, src_name)
    dst = os.path.join(static, dst_name)
    if not os.path.exists(src):
      print(f"  skip   {dst_name} (no source {src})")
      continue
    if os.path.exists(dst) and not os.path.exists(dst + ".orig"):
      shutil.copy2(dst, dst + ".orig")
    shutil.copy2(src, dst)
    print(f"  copied {src_name} -> {dst_name} ({_digest(dst)})")

  runtime_name, main_name, bundle_name = CONCAT_BUNDLE
  runtime_p = os.path.join(dist, runtime_name)
  main_p = os.path.join(dist, main_name)
  if not os.path.exists(main_p):
    sys.exit(f"Missing {main_p}")
  blob = b""
  if os.path.exists(runtime_p):
    with open(runtime_p, "rb") as f:
      blob += f.read() + b"\n"
  with open(main_p, "rb") as f:
    blob += f.read()

  dst = os.path.join(static, bundle_name)
  if os.path.exists(dst) and not os.path.exists(dst + ".orig"):
    shutil.copy2(dst, dst + ".orig")
  with open(dst, "wb") as f:
    f.write(blob)
  digest = hashlib.sha256(blob).hexdigest()
  with open(os.path.join(dist, BUNDLE_DIGEST_FILE), "w") as f:
    f.write(digest)
  print(f"  wrote  {runtime_name}+{main_name} -> {bundle_name} ({digest[:12]})")

  print(f"\nOverlaid this checkout's frontend onto {static}")
  print("Undo with: python -m tests.ui.sync_frontend --restore")
  return 0


def restore() -> int:
  """Restores original frontend assets from .orig backups."""
  static = served_static_dir()
  n = 0
  for name in os.listdir(static):
    if not name.endswith(".orig"):
      continue
    shutil.move(
        os.path.join(static, name),
        os.path.join(static, name[: -len(".orig")]),
    )
    print(f"  restored {name[: -len('.orig')]}")
    n += 1
  print(f"\nRestored {n} file(s) in {static}")
  return 0


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      "--restore", action="store_true", help="undo a previous overlay"
  )
  args = ap.parse_args()
  sys.exit(restore() if args.restore else overlay())
