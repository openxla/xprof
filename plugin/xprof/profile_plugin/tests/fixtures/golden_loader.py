# Copyright 2026 The XProf Authors. All Rights Reserved.
"""Load committed golden JSON snapshots."""

from __future__ import annotations

import json
from pathlib import Path


_GOLDEN_DIR = Path(__file__).resolve().parent / 'golden'


def load_golden(name: str) -> dict:
  """Load ``fixtures/golden/<name>.json`` (name without .json)."""
  path = _GOLDEN_DIR / f'{name}.json'
  with path.open('r', encoding='utf-8') as f:
    return json.load(f)
