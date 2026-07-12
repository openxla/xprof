/**
 * WASM heap ownership helpers for Trace Viewer v2.
 *
 * ## Free conventions
 *
 * 1. **Raw `_malloc` ownership (this file):** Any pointer obtained via
 *    `module._malloc` MUST be released with `module._free` after the native
 *    call returns. Prefer `withWasmHeapBuffer` / `withWasmCString` so free is
 *    guaranteed even if the callback throws. Never free a null/0 pointer.
 *
 * 2. **Embind / `emscripten::val` transfers:** JS strings and objects passed
 *    through embind (e.g. `setSearchResultsInWasm(traceData)`) are owned by the
 *    JavaScript GC. Do **not** call `_free` on values returned from embind
 *    string getters unless the C++ binding documentation explicitly says the
 *    pointer was allocated with `malloc` for JS to free.
 *
 * 3. **Search result transfer:** Prefer pointer+size heap transfers for large
 *    binary/protobuf payloads (`setCompressedSearchResultsInWasm`,
 *    `processCompressedTraceEvents`, `processPerfettoTraceEvents`) using
 *    `withWasmHeapBuffer`. JSON search results may continue to use embind.
 *
 * These helpers exist so new call sites do not reintroduce the #2753-class
 * leaks fixed for binary buffer transfers.
 */

/** Minimal module surface required for heap allocate/free. */
export interface WasmHeapModule {
  HEAPU8: Uint8Array;
  _malloc(size: number): number;
  _free(ptr: number): void;
}

/**
 * Null-safe free for a WASM heap pointer obtained via `_malloc`.
 * No-ops for undefined, null, or 0.
 */
export function freeWasmPtr(
  module: WasmHeapModule,
  ptr: number | null | undefined,
): void {
  if (ptr !== undefined && ptr !== null && ptr !== 0) {
    module._free(ptr);
  }
}

/**
 * Allocates `byteLength` bytes on the WASM heap, invokes `fn(ptr)`, and always
 * frees the pointer afterward (including when `fn` throws).
 *
 * Prefer `withWasmHeapBuffer` when the allocation is immediately filled from
 * a typed array.
 */
export function withWasmMalloc<T>(
  module: WasmHeapModule,
  byteLength: number,
  fn: (ptr: number) => T,
): T {
  if (byteLength < 0) {
    throw new Error('withWasmMalloc: byteLength must be non-negative');
  }
  // Zero-length allocations are valid but need no heap traffic.
  if (byteLength === 0) {
    return fn(0);
  }
  const ptr = module._malloc(byteLength);
  if (!ptr) {
    throw new Error('Failed to allocate WASM memory buffer');
  }
  try {
    return fn(ptr);
  } finally {
    freeWasmPtr(module, ptr);
  }
}

/**
 * Copies `data` into a freshly malloc'd WASM heap buffer, calls
 * `fn(ptr, byteLength)`, then always frees the buffer.
 *
 * Use this for binary transfers into exported C/C++ entry points that accept
 * `(dataPtr, dataSize)` (compressed traces, Perfetto, compressed search).
 */
export function withWasmHeapBuffer<T>(
  module: WasmHeapModule,
  data: Uint8Array | ArrayBuffer,
  fn: (ptr: number, byteLength: number) => T,
): T {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  return withWasmMalloc(module, bytes.byteLength, (ptr) => {
    if (bytes.byteLength > 0) {
      module.HEAPU8.set(bytes, ptr);
    }
    return fn(ptr, bytes.byteLength);
  });
}

/**
 * Encodes `jsString` as a null-terminated UTF-8 C string on the WASM heap,
 * invokes `fn(ptr)`, and always frees the allocation.
 *
 * Prefer embind for ordinary string arguments. Use this only when calling a
 * raw C API that expects `const char*`.
 */
export function withWasmCString<T>(
  module: WasmHeapModule,
  jsString: string,
  fn: (ptr: number) => T,
): T {
  // TextEncoder does not emit a trailing NUL; allocate one extra byte.
  const encoded = new TextEncoder().encode(jsString);
  const byteLength = encoded.length + 1;
  return withWasmMalloc(module, byteLength, (ptr) => {
    module.HEAPU8.set(encoded, ptr);
    module.HEAPU8[ptr + encoded.length] = 0;
    return fn(ptr);
  });
}
