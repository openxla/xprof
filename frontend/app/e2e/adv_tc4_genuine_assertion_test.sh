#!/bin/bash
set -euo pipefail

echo "=== Adversarial Technical Contract Verification ==="

MAIN_TS="third_party/xprof/frontend/app/components/trace_viewer_v2/main.ts"
HELPERS_TS="third_party/xprof/frontend/app/e2e/e2e_test_helpers.ts"
TIER3_TS="third_party/xprof/frontend/app/e2e/tier3_pairwise_e2e_test.ts"
TIER4_TS="third_party/xprof/frontend/app/e2e/tier4_real_world_e2e_test.ts"

# 1. Check TypeScript Strictness (zero `as any` or `as unknown` casting)
echo "Checking TypeScript strictness..."
for file in "$MAIN_TS" "$HELPERS_TS" "$TIER3_TS" "$TIER4_TS"; do
  if grep -q "as any" "$file"; then
    echo "FAIL: Found 'as any' in $file"
    exit 1
  fi
  if grep -q "as unknown" "$file"; then
    echo "FAIL: Found 'as unknown' in $file"
    exit 1
  fi
done
echo "PASS: 100% TypeScript strictness confirmed."

# 2. Check Immutable Mock Protection
echo "Checking Immutable Mock Protection..."
if ! grep -q "set: () => {}" "$HELPERS_TS"; then
  echo "FAIL: Immutable mock protection (set: () => {}) missing in $HELPERS_TS"
  exit 1
fi
echo "PASS: Immutable mock protection confirmed."

# 3. Check WASM Heap Bounds (65536)
echo "Checking WASM Heap Bounds..."
if ! grep -q "new Uint8Array(65536)" "$HELPERS_TS"; then
  echo "FAIL: WASM heap bounds (65536) missing in $HELPERS_TS"
  exit 1
fi
echo "PASS: WASM heap bounds confirmed."

# 4. Check Genuine _isProcessingFile Deduplication
echo "Checking Genuine _isProcessingFile Deduplication..."
if ! grep -q "if (traceviewerModule._isProcessingFile === file.name) return;" "$MAIN_TS"; then
  echo "FAIL: _isProcessingFile deduplication missing in $MAIN_TS"
  exit 1
fi
echo "PASS: Genuine _isProcessingFile deduplication confirmed."

# 5. Check Genuine Assertions in Tier 3 & Tier 4
echo "Checking Genuine Assertions in Tier 3 & Tier 4..."
if ! grep -q "expect(statusEvents).toEqual(\['Loading data', 'Processing data', 'Idle'\])" "$TIER3_TS"; then
  echo "FAIL: Genuine statusEvents assertion missing in $TIER3_TS"
  exit 1
fi
if ! grep -q "expect(statusEvents).toEqual(\['Loading data', 'Processing data', 'Idle'\])" "$TIER4_TS"; then
  echo "FAIL: Genuine statusEvents assertion missing in $TIER4_TS"
  exit 1
fi
if ! grep -q "file.arrayBuffer = () => Promise.reject(new Error('Corrupt trace file'))" "$TIER4_TS"; then
  echo "FAIL: Genuine rejected promise override missing in $TIER4_TS"
  exit 1
fi
if ! grep -q "expect(errorEventReceived).toBeTrue()" "$TIER4_TS"; then
  echo "FAIL: Genuine errorEventReceived assertion missing in $TIER4_TS"
  exit 1
fi
echo "PASS: Genuine assertions confirmed."

echo "=== All Adversarial Checks Passed Successfully ==="
