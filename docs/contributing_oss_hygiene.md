# OSS contribution and export hygiene

Process checklist for contributors and for first-party (1P) → open-source
exports into [openxla/xprof](https://github.com/openxla/xprof). These rules
exist so review stays focused on behavior, not noise, and so OSS consumers get
honest commit history.

## 1. Split dependency logic from lockfile regeneration

**Lesson:**
[PR #2823](https://github.com/openxla/xprof/pull/2823) mixed a useful GCS /
dependency fix with multi-thousand-line lockfile churn
(`requirements_lock_3_*.txt`). Reviewers could not see the real logic change.

### Rules

1. **Separate PRs (preferred)** or at least **separate commits**:
   - PR/commit A: `requirements.in` / `plugin/setup.py` / import-site code
     changes only (e.g. drop a dependency, tighten exception types).
   - PR/commit B: pure lockfile regeneration for all supported Python versions
     (`requirements_lock_3_10.txt` … `requirements_lock_3_13.txt`).
2. **Never** bury behavioral changes under lockfile fog. If a reviewer cannot
   `git show` the logic diff without paging through hashes, split the change.
3. Lock-only PRs should say so in the title, for example:
   `deps: regenerate lockfiles after removing pyOpenSSL`.
4. When both must land together for green CI, put the logic commit first and the
   lock regen second so `git log -p -1` on the logic commit stays reviewable.

### Checklist before opening a dependency PR

- [ ] Logic / `requirements.in` / setup metadata is reviewable without lockfiles.
- [ ] Lockfiles, if present, are a dedicated commit or PR.
- [ ] Title describes the user-visible change, not only “update deps”.

## 2. XLA / WORKSPACE pin bumps: smoke checklist

XProf pins OpenXLA via the `@xla` `http_archive` in [`WORKSPACE`](../WORKSPACE)
(`strip_prefix` / commit SHA / `sha256` / `//third_party:xla.patch`). Pin bumps
change protos, schemas, and C++ APIs that convert paths depend on. Silent pin
moves without smoke tests have broken convert tools in the past (see XLA pin
landings such as #2932 / #2849).

### Required PR description fields

- Old commit SHA → new commit SHA (and updated `sha256`).
- Why the pin moved (schema field, bugfix, toolchain, etc.).
- Notes on any XProf code or `third_party/xla.patch` updates required by the pin.

### Smoke checklist (run before merge)

From a clean tree after the pin change:

```bash
# Resolve the new archive and refresh the workspace.
bazel sync --only=xla

# Core convert / processor unit tests (adjust targets if names move).
bazel test //xprof/convert:op_stats_to_overview_page_test \
           //xprof/convert:xplane_to_op_stats_test \
           //xprof/convert:xplane_to_memory_profile_test \
           //xprof/convert:unified_overview_page_processor_test \
           //xprof/convert:unified_hlo_stats_processor_test \
           //xprof/convert:unified_memory_viewer_processor_test \
           //xprof/convert:unified_op_profile_processor_test

# Broader convert suite when the pin touches schema or HLO heavily.
bazel test //xprof/convert/...
```

Optional but recommended when UI or pip packaging might be affected:

```bash
bazel run --config=public_cache plugin:build_pip_package
```

### Checklist

- [ ] `WORKSPACE` SHA + `sha256` + `strip_prefix` agree.
- [ ] `third_party/xla.patch` still applies (or was updated intentionally).
- [ ] Convert smoke targets above are green.
- [ ] PR body lists old/new pin and schema notes.

## 3. Protobuf compatibility bounds (soft docs)

Runtime protobuf version checks were removed
([PR #2854](https://github.com/openxla/xprof/pull/2854)) so installs are less
brittle, but mismatched `protobuf` can still produce cryptic decode errors.

### Declared floor

- **Minimum:** `protobuf >= 3.19.6` (see [`requirements.in`](../requirements.in)
  and [`plugin/setup.py`](../plugin/setup.py)).
- **Lockfiles:** currently pin a modern `protobuf` (see
  `requirements_lock_3_*.txt` for the exact version CI/builds use).

### Soft guidance for developers and packagers

1. Prefer the lockfile version when building from source.
2. If you must use a system or environment `protobuf`, stay **≥ 3.19.6** and
   ideally near the lockfile major when reading XProf-generated `*_pb2.py`.
3. Pip packaging strips generated runtime version guards in
   `plugin/build_pip_package.sh` so wheels install on a wider range of
   environments; that is intentional, not a license to ignore the floor.
4. On mysterious profile parse failures after a dependency upgrade, check
   `python -c "import google.protobuf; print(google.protobuf.__version__)"`
   before debugging XProf itself.

This document is the soft warning; hard runtime asserts are intentionally not
re-introduced.

## 4. OSS export / Copybara commit hygiene

Opaque 1P exports create unreviewable history on GitHub (“Internal change”,
“No op”, “Project import”, Piper-only landings without a PR). Examples include
#2954 / #2832 (empty “Internal change” subjects) and Piper commits such as
`76770791`, `c4480496`, `dd30d855` that never opened a GitHub PR.

### Ban empty or 1P-only subjects

**Do not** use these as the sole GitHub subject:

- `Internal change`
- `No op` / `No-op` / `NOOP`
- `Project import`
- Empty or whitespace-only subjects

Every export that lands on `openxla/xprof` must have a **subject that states the
OSS-visible intent**, for example:

- `fix(cli): document --hide_capture_profile_button default`
- `refactor(trace_viewer): convert PrefixTrie build to iterative DFS`

If the internal description is confidential, rewrite a public rationale; do not
ship a placeholder.

### Strip pure 1P no-ops

Do not export commits that only touch:

- Internal-only paths that do not exist in the OSS layout
- No-op reformatting or metadata with zero OSS behavioral or docs impact
- Add/delete thrash of the same blob across consecutive exports (squash to one
  intentional landing before export)

### Always open a GitHub PR for Copybara / Piper exports

1. Every OSS-visible change should have a **GitHub PR** on `openxla/xprof`
   (or a clearly linked PR from the export pipeline).
2. Prefer **one logical concern per PR**; do not batch unrelated convert, UI, and
   lockfile edits solely because they shared a Piper CL.
3. PR body should include: summary, test plan, and any follow-ups. “Internal
   change” is not a body.
4. If automation merges without review, still ensure the generated PR title and
   description meet the rules above so `git log` and the GitHub UI remain useful.

### Export preflight checklist

- [ ] Subject is meaningful to an external reader.
- [ ] No pure 1P no-op commits in the export batch.
- [ ] GitHub PR exists and is linked from the export metadata.
- [ ] Lockfiles / XLA pins follow sections 1–2 if present in the same batch.
- [ ] No stray `-->` leftovers (section 5).

## 5. Stray HTML comment terminators (`-->`)

**Lesson:** [PR #2828](https://github.com/openxla/xprof/pull/2828) removed a
literal `-->` left in an Angular template after a bad 1P/HTML comment edit.
Stray `-->` outside a real comment breaks templates and confuses diffs.

### Lint note (local / CI)

Before exporting frontend or plugin HTML/templates, search for comment
terminator leftovers:

```bash
# Flag lines that are only "-->" or clearly stray terminators in templates.
# Tune paths as the tree evolves.
git grep -nE '^\s*-->\s*$|<!--.*$|^[^<]*-->' -- \
  'frontend/**/*.html' \
  'frontend/**/*.ng.html' \
  'plugin/**/*.html' \
  'labs/**/*.html' || true

# Narrow check: lines that are solely a closing comment token (common footgun).
git grep -nE '^\s*-->\s*$' -- frontend plugin labs || true
```

### Rules

1. HTML comments must be well-formed: `<!-- … -->` on the same logical edit.
2. Do not leave a lone `-->` after deleting the opening `<!--` or the comment body.
3. Prefer component/TS flags over large commented-out HTML blocks in templates.
4. If adding CI later, fail the build on `^\s*-->\s*$` under `frontend/` and
   `plugin/` template globs.

## Quick reference

| Topic | Do | Don't |
|-------|----|--------|
| Dependencies | Split logic vs lock regen (#2823) | One mega-PR of +4k/−3k lock noise |
| XLA pins | Smoke convert tests + changelog | Bump SHA with no test plan |
| Protobuf | Document ≥ 3.19.6; prefer lockfile | Assume any protobuf works |
| Export subjects | Describe OSS intent | `Internal change` only |
| Copybara | Open GH PR; strip 1P no-ops | Piper-only landings with no PR |
| Templates | Grep for stray `-->` | Ship broken comment leftovers |

## See also

- User-facing guides under [`docs/`](./)
- Build-from-source notes in the root [`README.md`](../README.md)
