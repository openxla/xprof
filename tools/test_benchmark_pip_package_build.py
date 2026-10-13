#!/usr/bin/env python3
"""Tests for tools/benchmark_pip_package_build.sh (real script path, no mocks)."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "benchmark_pip_package_build.sh"
PIP_BUILD_SH = ROOT / "plugin" / "build_pip_package.sh"
PLUGIN_BUILD = ROOT / "plugin" / "BUILD"


class BenchmarkPipPackageBuildScriptTest(unittest.TestCase):
    def test_script_and_shipped_entrypoint_exist(self):
        self.assertTrue(SCRIPT.is_file(), f"missing {SCRIPT}")
        self.assertTrue(os.access(SCRIPT, os.X_OK), f"not executable: {SCRIPT}")
        self.assertTrue(PIP_BUILD_SH.is_file(), f"missing {PIP_BUILD_SH}")
        build_text = PLUGIN_BUILD.read_text(encoding="utf-8")
        self.assertIn('name = "build_pip_package"', build_text)
        self.assertIn("build_pip_package.sh", build_text)

    def test_script_invokes_real_target_and_clean(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('TARGET="plugin:build_pip_package"', text)
        self.assertIn('"$BAZEL" run', text)
        self.assertIn('"${TARGET}"', text)
        self.assertIn("bazel clean", text)
        self.assertIn("DURATION_SEC", text)
        self.assertIn("EXIT_CODE", text)
        self.assertIn("do_clean", text)
        # Must not be a stub that skips bazel run entirely for matrix mode.
        self.assertNotRegex(text, r"echo\s+STUB.*build_pip_package")

    def test_list_prints_distinct_configs(self):
        proc = subprocess.run(
            [str(SCRIPT), "--list"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 5, proc.stdout)
        names = []
        flags_set = set()
        for ln in lines:
            self.assertIn(" | ", ln, ln)
            name, flags = ln.split(" | ", 1)
            names.append(name.strip())
            flags_set.add(flags.strip())
        self.assertEqual(len(names), len(set(names)), "config names must be unique")
        self.assertGreaterEqual(len(flags_set), 4, "need distinct flag texts")
        # Baseline axis present
        joined = "\n".join(lines)
        self.assertIn("public_cache", joined)

    def test_dry_run_writes_structured_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["LOG_DIR"] = tmp
            proc = subprocess.run(
                [str(SCRIPT), "--dry-run"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            self.assertIn("plugin:build_pip_package", proc.stdout)
            logs = list(Path(tmp).glob("bazel_pip_benchmark_*.log"))
            tsvs = list(Path(tmp).glob("attempts_*.tsv"))
            self.assertEqual(len(logs), 1, logs)
            self.assertEqual(len(tsvs), 1, tsvs)
            log_text = logs[0].read_text(encoding="utf-8")
            self.assertIn("CLEAN", log_text.upper() + log_text)
            self.assertIn("DRY_RUN", log_text)
            tsv = tsvs[0].read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(tsv), 6)  # header + >=5
            header = tsv[0].split("\t")
            self.assertIn("duration_sec", header)
            self.assertIn("exit_code", header)
            self.assertIn("flags", header)
            flag_cells = set()
            for row in tsv[1:]:
                parts = row.split("\t")
                self.assertGreaterEqual(len(parts), 5, row)
                flag_cells.add(parts[4])
            self.assertGreaterEqual(len(flag_cells), 4)



    def test_xcode_config_for_clt_exists(self):
        xcode_build = ROOT / "tools" / "xcode" / "BUILD"
        self.assertTrue(xcode_build.is_file(), xcode_build)
        xb = xcode_build.read_text(encoding="utf-8")
        self.assertIn('name = "host_xcodes"', xb)
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("xcode_version_config=//tools/xcode:host_xcodes", script)
        self.assertIn("BASE_FLAGS", script)

if __name__ == "__main__":
    unittest.main()
