export type SafeScript = string;
const safeScript = (s: TemplateStringsArray) => s[0];

/** Downloads the trace at a URL, like `fetch`. */
export type TraceDataFetcher = (url: string) => Promise<Response>;

/**
 * Constructs the dedicated Web Worker used to stream and cache compressed trace
 * payloads. Accepts the allocated Blob URL and inline worker script so unit
 * tests can substitute an isolated worker harness without mutating module
 * globals.
 */
export type TraceStreamWorkerFactory = (
  workerBlobUrl: string,
  workerScript: SafeScript,
) => Worker;

/**
 * Request message sent from the main thread to the dedicated trace-streaming
 * Web Worker to initiate streaming (and optional OPFS caching) for a compressed
 * trace payload.
 *
 * Declared with `declare interface` so Closure Compiler preserves property
 * names across the `postMessage` boundary into the inline worker script.
 */
declare interface WorkerTraceRequest {
  /** Trace endpoint URL to fetch (including `stream=true`). */
  url: string;
  /** Canonical session-level cache key used to index OPFS cache entries. */
  cacheKey: string;
  /** When true, skips reading any cached trace entry from OPFS. */
  bypassCache: boolean;
  /** When true, stages and commits the completed full-session stream to OPFS. */
  persistCache: boolean;
}

/**
 * Response message posted from the dedicated trace-streaming Web Worker back to
 * the main thread as individual compressed trace frames arrive or when the
 * stream finishes or fails.
 */
declare interface WorkerTraceMessage {
  /** Message kind: an individual compressed frame, stream completion, or error. */
  type: 'frame' | 'done' | 'error';
  /** Transferred `ArrayBuffer` containing a single compressed trace frame. */
  buffer?: ArrayBuffer;
  /** Human-readable error message when `type === 'error'`. */
  message?: string;
}

/**
 * Options for `streamCompressedTraceFrames`.
 */
export interface StreamCompressedTraceFramesOptions {
  urlObj: URL;
  cacheKey: string;
  bypassCache: boolean;
  persistCache: boolean;
  onFrame: (frame: Uint8Array) => void;
  isAbortRequested: () => boolean;
  abortSignal: AbortSignal;
  fetchData?: TraceDataFetcher;
  createStreamWorker?: TraceStreamWorkerFactory;
}

/**
 * Returns true when client-side cache bypass is explicitly requested via
 * `nocache=true` or `nocache=1`.
 */
function isClientCacheBypassRequested(params: URLSearchParams): boolean {
  const nocache = params.get('nocache')?.trim().toLowerCase();
  return nocache === 'true' || nocache === '1';
}

/**
 * Returns true when a backend cache refresh is requested via
 * `use_saved_result=0` or `use_saved_result=false` (inverted relative to
 * `nocache=true`).
 */
function isBackendCacheRefreshRequested(params: URLSearchParams): boolean {
  const useSavedResult = params
    .get('use_saved_result')
    ?.trim()
    .toLowerCase();
  return useSavedResult === '0' || useSavedResult === 'false';
}

/**
 * Returns true when client-side OPFS cache bypass (`nocache=true` or `1`) is
 * requested on either `urlObj` or `window.location.search`.
 */
export function isClientOpfsBypassRequested(urlObj: URL): boolean {
  return (
    isClientCacheBypassRequested(urlObj.searchParams) ||
    (typeof window !== 'undefined' &&
      isClientCacheBypassRequested(new URLSearchParams(window.location.search)))
  );
}

/**
 * Determines whether reading from the client-side OPFS trace cache (layer 1)
 * should be bypassed based on two-layer cache parameters on the trace URL or
 * browser location:
 * - Client-side bypass (`nocache=true` / `1`) skips both reading from and
 *   writing to OPFS.
 * - Backend cache refresh (`use_saved_result=0` / `false`, layer 2) forces a
 *   fresh backend conversion and skips reading stale OPFS data, while still
 *   allowing full-session loads (when `nocache` is not set) to update OPFS with
 *   the newly regenerated trace.
 */
export function shouldBypassOpfsCache(urlObj: URL): boolean {
  const shouldBypassForParams = (params: URLSearchParams) =>
    isClientCacheBypassRequested(params) ||
    isBackendCacheRefreshRequested(params);

  return (
    shouldBypassForParams(urlObj.searchParams) ||
    (typeof window !== 'undefined' &&
      shouldBypassForParams(new URLSearchParams(window.location.search)))
  );
}

/** Returns a user-facing error message for a non-OK trace `Response`. */
export function getFetchErrorMessage(response: Response): string {
  if (response.status === 404) {
    return 'Trace data not found (404). The profile session may have expired or does not contain trace data.';
  }
  return `Failed to load trace data: HTTP error! status: ${response.status}`;
}

/** Returns true when `error` represents an aborted fetch (`AbortError`). */
export function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException || error instanceof Error) &&
    error.name === 'AbortError'
  );
}

/**
 * Fetches a compressed trace payload into a single `ArrayBuffer` on the main
 * thread (used when a custom `fetchData` override is supplied or when Web
 * Workers are unavailable or blocked by CSP). When no custom `fetchData` is
 * provided, forwards `abortSignal` to `fetch` so session switches or
 * cancellations immediately abort an in-flight download.
 */
export async function loadCompressedTraceDataInternal(
  url: string,
  fetchData?: TraceDataFetcher,
  abortSignal?: AbortSignal,
): Promise<ArrayBuffer> {
  try {
    const response = fetchData
      ? await fetchData(url)
      : abortSignal
        ? await fetch(url, {signal: abortSignal})
        : await fetch(url);
    if (!response.ok) {
      throw new Error(getFetchErrorMessage(response));
    }
    return await response.arrayBuffer();
  } catch (e) {
    if (!isAbortError(e)) {
      console.error('Failed to load compressed trace data:', e);
    }
    throw e;
  }
}

/**
 * Splits a length-prefixed `XCHN` framing payload (`"XCHN"` + repeated
 * `[uint32_le length][payload]` terminated by `[uint32_le 0]`) into individual
 * frame slices, or returns the raw payload as a single frame when the `XCHN`
 * header is not present.
 */
export function extractCompressedFrames(buffer: ArrayBuffer): Uint8Array[] {
  const bytes = new Uint8Array(buffer);
  if (
    bytes.byteLength >= 4 &&
    bytes[0] === 0x58 &&
    bytes[1] === 0x43 &&
    bytes[2] === 0x48 &&
    bytes[3] === 0x4e
  ) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const frames: Uint8Array[] = [];
    let offset = 4;
    let sawEndMarker = false;
    while (offset + 4 <= bytes.byteLength) {
      const frameLen = view.getUint32(offset, true);
      offset += 4;
      if (frameLen === 0) {
        sawEndMarker = true;
        break;
      }
      if (frameLen > bytes.byteLength - offset) break;
      frames.push(bytes.subarray(offset, offset + frameLen));
      offset += frameLen;
    }
    if (!sawEndMarker) {
      throw new Error('Trace stream ended unexpectedly before completion.');
    }
    return frames;
  }
  return bytes.byteLength > 0 ? [bytes] : [];
}

function createTraceStreamWorker(
  createStreamWorker?: TraceStreamWorkerFactory,
): {
  worker: Worker;
  revokeUrl: () => void;
} | null {
  if (!createStreamWorker && typeof Worker === 'undefined') {
    return null;
  }
  let workerUrlToRevoke: string | null = null;
  try {
    // TODO(xprof): Compile this worker from a standalone .ts entrypoint
    // (bundled into a generated safeScript module) to get compile-time type
    // safety, direct unit testability, and shared XCHN framing/error helpers.
    const script = safeScript`
      self.onmessage = async (event) => {
        const data = event.data;
        const url = data && data.url;
        if (!url) return;
        const cacheKey = (data && data.cacheKey) || '';
        const bypassCache = Boolean(data && data.bypassCache);
        const persistCache = Boolean(data && data.persistCache);

        const hashCacheKey = (key) => {
          let h1 = 0x811c9dc5;
          let h2 = 0x01000193;
          for (let i = 0; i < key.length; i++) {
            const c = key.charCodeAt(i);
            h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
            h2 = Math.imul(h2 ^ (c + i), 0x85ebca6b) >>> 0;
          }
          return h1.toString(16).padStart(8, '0') + h2.toString(16).padStart(8, '0');
        };
        const keyHash = cacheKey ? hashCacheKey(cacheKey) : '';
        const metaFileName = keyHash ? 'trace_' + keyHash + '.meta.json' : '';

        let opfsRoot = null;
        if (cacheKey && (!bypassCache || persistCache)) {
          try {
            if (
              self.navigator &&
              self.navigator.storage &&
              typeof self.navigator.storage.getDirectory === 'function'
            ) {
              opfsRoot = await self.navigator.storage.getDirectory();
            }
          } catch (_) {
            opfsRoot = null;
          }
        }

        const openReadSyncHandle = async (fileHandle) => {
          try {
            return await fileHandle.createSyncAccessHandle({mode: 'read-only'});
          } catch (err) {
            if (err && err.name !== 'NoModificationAllowedError') {
              return await fileHandle.createSyncAccessHandle();
            }
            throw err;
          }
        };

        const readMetaHandleText = async (metaHandle) => {
          if (typeof metaHandle.getFile === 'function') {
            const metaFile = await metaHandle.getFile();
            if (metaFile.size > 0 && metaFile.size <= (16 << 20)) {
              return await metaFile.text();
            }
            return '';
          }
          if (typeof metaHandle.createSyncAccessHandle === 'function') {
            const metaSync = await openReadSyncHandle(metaHandle);
            try {
              const metaSize = metaSync.getSize();
              if (metaSize > 0 && metaSize <= (16 << 20)) {
                const metaBuf = new Uint8Array(metaSize);
                metaSync.read(metaBuf, {at: 0});
                return new TextDecoder().decode(metaBuf);
              }
            } finally {
              try { metaSync.close(); } catch (_) {}
            }
          }
          return '';
        };

        // Replay completed full-session stream from OPFS if cached for cacheKey.
        if (opfsRoot && metaFileName && !bypassCache) {
          let postedAnyFrame = false;
          try {
            const readDir =
              typeof opfsRoot.getDirectoryHandle === 'function'
                ? await opfsRoot.getDirectoryHandle('xprof_trace_cache', {create: false})
                : opfsRoot;
            const metaHandle = await readDir.getFileHandle(metaFileName, {create: false});
            const metaText = await readMetaHandleText(metaHandle);
            if (metaText) {
              const meta = JSON.parse(metaText);
              if (
                meta &&
                meta.complete === true &&
                meta.cacheKey === cacheKey &&
                typeof meta.dataFile === 'string' &&
                meta.dataFile.length > 0 &&
                Number.isInteger(meta.totalBytes) &&
                meta.totalBytes > 0 &&
                Array.isArray(meta.frames) &&
                meta.frames.length > 0
              ) {
                const dataHandle = await readDir.getFileHandle(meta.dataFile, {create: false});
                if (dataHandle && typeof dataHandle.createSyncAccessHandle === 'function') {
                  const dataSync = await openReadSyncHandle(dataHandle);
                  try {
                    const fileSize = dataSync.getSize();
                    const validFrames =
                      fileSize === meta.totalBytes &&
                      meta.frames.every(
                        (entry) =>
                          entry &&
                          Number.isInteger(entry.offset) &&
                          entry.offset >= 0 &&
                          Number.isInteger(entry.length) &&
                          entry.length > 0 &&
                          entry.offset + entry.length <= fileSize,
                      );
                    if (validFrames) {
                      for (let i = 0; i < meta.frames.length; i++) {
                        const entry = meta.frames[i];
                        const out = new Uint8Array(entry.length);
                        const bytesRead = dataSync.read(out, {at: entry.offset});
                        if (bytesRead !== entry.length) {
                          throw new Error('Failed to read complete cached trace frame from OPFS.');
                        }
                        postedAnyFrame = true;
                        self.postMessage({type: 'frame', buffer: out.buffer}, [out.buffer]);
                        if (i === 0 || (i & 15) === 15) {
                          await new Promise((r) => setTimeout(r, 0));
                        }
                      }
                      self.postMessage({type: 'done'});
                      return;
                    }
                  } finally {
                    try { dataSync.close(); } catch (_) {}
                  }
                }
              }
            }
          } catch (opfsReadErr) {
            if (postedAnyFrame) {
              self.postMessage({
                type: 'error',
                message:
                  opfsReadErr && typeof opfsReadErr.message === 'string'
                    ? opfsReadErr.message
                    : String(opfsReadErr),
              });
              return;
            }
            // Cache miss or unreadable OPFS metadata before any frame was emitted;
            // fall through to network fetch.
          }
        }

        let cacheDir = null;
        let stageFileName = '';
        let opfsSync = null;
        let opfsOffset = 0;
        let opfsWriteOk = false;
        const stagedFrames = [];

        const cleanupFailedOpfsStage = async () => {
          opfsWriteOk = false;
          if (opfsSync) {
            try { opfsSync.truncate(0); } catch (_) {}
            try { opfsSync.close(); } catch (_) {}
            opfsSync = null;
          }
          if (cacheDir && stageFileName) {
            const toRemove = stageFileName;
            stageFileName = '';
            try { await cacheDir.removeEntry(toRemove); } catch (_) {}
          }
        };

        const persistAndEmitFrame = (frameBytes) => {
          if (opfsSync && opfsWriteOk) {
            try {
              const written = opfsSync.write(frameBytes, {at: opfsOffset});
              if (written === frameBytes.byteLength) {
                stagedFrames.push({offset: opfsOffset, length: frameBytes.byteLength});
                opfsOffset += frameBytes.byteLength;
              } else {
                opfsWriteOk = false;
                try { opfsSync.truncate(0); } catch (_) {}
                try { opfsSync.close(); } catch (_) {}
                opfsSync = null;
              }
            } catch (_) {
              opfsWriteOk = false;
              try { opfsSync.truncate(0); } catch (_) {}
              try { opfsSync.close(); } catch (_) {}
              opfsSync = null;
            }
          }
          const frameBuffer = frameBytes.buffer;
          self.postMessage({type: 'frame', buffer: frameBuffer}, [frameBuffer]);
        };

        const finalizeOpfsCache = async () => {
          if (opfsSync) {
            try {
              opfsSync.flush();
            } catch (_) {
              opfsWriteOk = false;
            }
          }
          if (!cacheDir || !opfsWriteOk || !metaFileName || !stageFileName || stagedFrames.length === 0) {
            await cleanupFailedOpfsStage();
            return;
          }
          const nowMs = Date.now();
          let committed = false;
          try {
            const metaHandle = await cacheDir.getFileHandle(metaFileName, {create: true});
            if (typeof metaHandle.createSyncAccessHandle === 'function') {
              const metaSync = await metaHandle.createSyncAccessHandle();
              try {
                metaSync.truncate(0);
                const metaBytes = new TextEncoder().encode(JSON.stringify({
                  cacheKey,
                  complete: true,
                  dataFile: stageFileName,
                  savedAt: nowMs,
                  totalBytes: opfsOffset,
                  frames: stagedFrames,
                }));
                const written = metaSync.write(metaBytes, {at: 0});
                metaSync.flush();
                committed = written === metaBytes.byteLength;
              } finally {
                try { metaSync.close(); } catch (_) {}
              }
            }
          } catch (_) {
            committed = false;
          } finally {
            if (opfsSync) {
              try { opfsSync.close(); } catch (_) {}
              opfsSync = null;
            }
          }
          if (!committed) {
            await cleanupFailedOpfsStage();
            return;
          }
          if (typeof cacheDir.keys === 'function') {
            try {
              const allEntries = [];
              for await (const entryName of cacheDir.keys()) {
                allEntries.push(entryName);
              }
              const metaInfos = [];
              for (const entryName of allEntries) {
                if (!entryName.endsWith('.meta.json')) continue;
                if (entryName === metaFileName) {
                  metaInfos.push({metaFile: entryName, dataFile: stageFileName, savedAt: nowMs});
                  continue;
                }
                try {
                  const otherHandle = await cacheDir.getFileHandle(entryName, {create: false});
                  const otherText = await readMetaHandleText(otherHandle);
                  const parsed = otherText ? JSON.parse(otherText) : null;
                  if (parsed && typeof parsed.dataFile === 'string') {
                    metaInfos.push({
                      metaFile: entryName,
                      dataFile: parsed.dataFile,
                      savedAt: typeof parsed.savedAt === 'number' ? parsed.savedAt : 0,
                    });
                  } else {
                    try { await cacheDir.removeEntry(entryName); } catch (_) {}
                  }
                } catch (_) {}
              }
              metaInfos.sort((a, b) => b.savedAt - a.savedAt);
              const keepSet = new Set([metaFileName, stageFileName]);
              for (let i = 0; i < Math.min(2, metaInfos.length); i++) {
                keepSet.add(metaInfos[i].metaFile);
                keepSet.add(metaInfos[i].dataFile);
              }
              for (const entryName of allEntries) {
                if (!keepSet.has(entryName)) {
                  try { await cacheDir.removeEntry(entryName); } catch (_) {}
                }
              }
            } catch (_) {}
          }
        };

        try {
          const response = await fetch(url);
          if (!response.ok) {
            const errorMsg =
              response.status === 404
                ? 'Trace data not found (404). The profile session may have expired or does not contain trace data.'
                : 'Failed to load trace data: HTTP error! status: ' + response.status;
            self.postMessage({
              type: 'error',
              message: errorMsg,
            });
            return;
          }

          if (opfsRoot && metaFileName && persistCache) {
            try {
              cacheDir =
                typeof opfsRoot.getDirectoryHandle === 'function'
                  ? await opfsRoot.getDirectoryHandle('xprof_trace_cache', {create: true})
                  : opfsRoot;
              stageFileName =
                'trace_' +
                keyHash +
                '_' +
                Date.now().toString(36) +
                '_' +
                Math.random().toString(36).slice(2, 10) +
                '.xchn';
              const stageHandle = await cacheDir.getFileHandle(stageFileName, {create: true});
              if (typeof stageHandle.createSyncAccessHandle === 'function') {
                opfsSync = await stageHandle.createSyncAccessHandle();
                opfsSync.truncate(0);
                opfsWriteOk = true;
              } else {
                try { await cacheDir.removeEntry(stageFileName); } catch (_) {}
                stageFileName = '';
              }
            } catch (_) {
              await cleanupFailedOpfsStage();
            }
          }

          const isXchnHeader = (bytes) =>
            bytes.length >= 4 &&
            bytes[0] === 0x58 &&
            bytes[1] === 0x43 &&
            bytes[2] === 0x48 &&
            bytes[3] === 0x4e;
          const readUint32Le = (bytes, offset) =>
            (bytes[offset] |
              (bytes[offset + 1] << 8) |
              (bytes[offset + 2] << 16) |
              (bytes[offset + 3] << 24)) >>>
            0;
          if (!response.body || typeof response.body.getReader !== 'function') {
            const buffer = await response.arrayBuffer();
            const bytes = new Uint8Array(buffer);
            if (isXchnHeader(bytes)) {
              let offset = 4;
              let sawEndOfStream = false;
              while (offset + 4 <= bytes.length) {
                const frameLen = readUint32Le(bytes, offset);
                offset += 4;
                if (frameLen === 0) {
                  sawEndOfStream = true;
                  break;
                }
                if (offset + frameLen > bytes.length) break;
                const frameSlice = bytes.slice(offset, offset + frameLen);
                if (frameSlice.byteLength > 0) {
                  persistAndEmitFrame(frameSlice);
                }
                offset += frameLen;
              }
              if (!sawEndOfStream) {
                await cleanupFailedOpfsStage();
                self.postMessage({
                  type: 'error',
                  message: 'Trace stream ended unexpectedly before completion.',
                });
                return;
              }
            } else if (bytes.byteLength > 0) {
              persistAndEmitFrame(bytes);
            }
            await finalizeOpfsCache();
            self.postMessage({type: 'done'});
            return;
          }

          const reader = response.body.getReader();
          const chunks = [];
          let headIdx = 0;
          let headOffset = 0;
          let queuedBytes = 0;
          const pushChunk = (chunk) => {
            chunks.push(chunk);
            queuedBytes += chunk.byteLength;
          };
          const peekByte = (relOffset) => {
            let idx = headIdx;
            let off = headOffset + relOffset;
            while (idx < chunks.length) {
              const c = chunks[idx];
              if (off < c.byteLength) {
                return c[off];
              }
              off -= c.byteLength;
              idx++;
            }
            return 0;
          };
          const peekUint32Le = () =>
            (peekByte(0) |
              (peekByte(1) << 8) |
              (peekByte(2) << 16) |
              (peekByte(3) << 24)) >>>
            0;
          const discardBytes = (count) => {
            let rem = count;
            queuedBytes -= count;
            while (rem > 0 && headIdx < chunks.length) {
              const avail = chunks[headIdx].byteLength - headOffset;
              if (rem < avail) {
                headOffset += rem;
                rem = 0;
              } else {
                rem -= avail;
                headIdx++;
                headOffset = 0;
              }
            }
            if (headIdx >= 64 && headIdx * 2 >= chunks.length) {
              chunks.splice(0, headIdx);
              headIdx = 0;
            }
          };
          const consumeBytes = (count) => {
            const out = new Uint8Array(count);
            let dstOffset = 0;
            let rem = count;
            queuedBytes -= count;
            while (rem > 0 && headIdx < chunks.length) {
              const c = chunks[headIdx];
              const avail = c.byteLength - headOffset;
              const take = rem < avail ? rem : avail;
              out.set(c.subarray(headOffset, headOffset + take), dstOffset);
              dstOffset += take;
              rem -= take;
              if (take === avail) {
                headIdx++;
                headOffset = 0;
              } else {
                headOffset += take;
              }
            }
            if (headIdx >= 64 && headIdx * 2 >= chunks.length) {
              chunks.splice(0, headIdx);
              headIdx = 0;
            }
            return out;
          };

          let mode = 'init';
          let sawEndOfStream = false;
          while (true) {
            const {done, value} = await reader.read();
            if (value && value.byteLength > 0) {
              pushChunk(value);
            }
            if (mode === 'init' && (queuedBytes >= 4 || done)) {
              if (
                queuedBytes >= 4 &&
                peekByte(0) === 0x58 &&
                peekByte(1) === 0x43 &&
                peekByte(2) === 0x48 &&
                peekByte(3) === 0x4e
              ) {
                mode = 'xchn';
                discardBytes(4);
              } else {
                mode = 'raw';
              }
            }
            if (mode === 'xchn') {
              while (!sawEndOfStream && queuedBytes >= 4) {
                const frameLen = peekUint32Le();
                if (frameLen === 0) {
                  sawEndOfStream = true;
                  discardBytes(4);
                  break;
                }
                if (queuedBytes < 4 + frameLen) {
                  break;
                }
                discardBytes(4);
                const frameBytes = consumeBytes(frameLen);
                if (frameBytes.byteLength > 0) {
                  persistAndEmitFrame(frameBytes);
                }
              }
            }
            if (done) {
              if (mode === 'xchn' && !sawEndOfStream) {
                await cleanupFailedOpfsStage();
                self.postMessage({
                  type: 'error',
                  message: 'Trace stream ended unexpectedly before completion.',
                });
                break;
              }
              if (mode === 'raw' && queuedBytes > 0) {
                persistAndEmitFrame(consumeBytes(queuedBytes));
              }
              await finalizeOpfsCache();
              self.postMessage({type: 'done'});
              break;
            }
          }
        } catch (e) {
          await cleanupFailedOpfsStage();
          self.postMessage({
            type: 'error',
            message:
              e && typeof e.message === 'string' ? e.message : String(e),
          });
        }
      };
    `;
    const workerUrl = URL.createObjectURL(
      new Blob([script], {type: 'application/javascript'}),
    );
    workerUrlToRevoke = workerUrl;
    const worker = createStreamWorker
      ? createStreamWorker(workerUrlToRevoke, script)
      : new Worker(workerUrl);
    const urlToRevoke = workerUrlToRevoke;
    let revoked = false;
    return {
      worker,
      revokeUrl: () => {
        if (!revoked) {
          revoked = true;
          URL.revokeObjectURL(urlToRevoke);
        }
      },
    };
  } catch {
    if (workerUrlToRevoke !== null) {
      URL.revokeObjectURL(workerUrlToRevoke);
    }
    return null;
  }
}

/**
 * Streams compressed trace frames from `urlObj` via a dedicated Web Worker
 * (with OPFS session cache support when `bypassCache` is false or
 * `persistCache` is true), falling back to main-thread `fetch` when `fetchData`
 * is supplied or when Web Worker creation/execution is blocked.
 */
export async function streamCompressedTraceFrames({
  urlObj,
  cacheKey,
  bypassCache,
  persistCache,
  onFrame,
  isAbortRequested,
  abortSignal,
  fetchData,
  createStreamWorker,
}: StreamCompressedTraceFramesOptions): Promise<void> {
  const fallbackToMainThreadFetch = async (): Promise<void> => {
    let buffer: ArrayBuffer;
    try {
      buffer = await loadCompressedTraceDataInternal(
        urlObj.toString(),
        fetchData,
        abortSignal,
      );
    } catch (e) {
      if (isAbortError(e) || isAbortRequested() || abortSignal.aborted) {
        return;
      }
      throw e;
    }
    if (isAbortRequested() || abortSignal.aborted) {
      return;
    }
    for (const frame of extractCompressedFrames(buffer)) {
      if (isAbortRequested() || abortSignal.aborted) {
        return;
      }
      if (frame.byteLength > 0) {
        onFrame(frame);
      }
    }
  };

  if (isAbortRequested() || abortSignal.aborted) {
    return;
  }

  const workerHandle = fetchData
    ? null
    : createTraceStreamWorker(createStreamWorker);
  if (!workerHandle) {
    await fallbackToMainThreadFetch();
    return;
  }

  const streamUrlObj = new URL(urlObj.toString());
  if (!streamUrlObj.searchParams.has('stream')) {
    streamUrlObj.searchParams.set('stream', 'true');
  }

  return new Promise<void>((resolve, reject) => {
    const {worker, revokeUrl} = workerHandle;
    let cleanedUp = false;
    let receivedAnyFrame = false;
    const onAbort = () => {
      cleanup();
      resolve();
    };
    const cleanup = () => {
      if (cleanedUp) return;
      cleanedUp = true;
      abortSignal.removeEventListener('abort', onAbort);
      worker.terminate();
      revokeUrl();
    };
    if (abortSignal.aborted || isAbortRequested()) {
      cleanup();
      resolve();
      return;
    }
    abortSignal.addEventListener('abort', onAbort, {once: true});

    worker.onmessage = (event: MessageEvent<WorkerTraceMessage>) => {
      if (isAbortRequested() || abortSignal.aborted) {
        cleanup();
        resolve();
        return;
      }
      const data = event.data;
      if (!data) return;
      if (data.type === 'frame' && data.buffer) {
        receivedAnyFrame = true;
        try {
          onFrame(new Uint8Array(data.buffer));
        } catch (err: unknown) {
          cleanup();
          reject(err instanceof Error ? err : new Error(String(err)));
        }
      } else if (data.type === 'done') {
        cleanup();
        resolve();
      } else if (data.type === 'error') {
        cleanup();
        const err = new Error(
          data.message || 'Worker failed to load compressed trace data',
        );
        console.error('Failed to load compressed trace data:', err);
        reject(err);
      }
    };
    worker.onerror = (event: ErrorEvent) => {
      cleanup();
      if (!receivedAnyFrame && !isAbortRequested() && !abortSignal.aborted) {
        event.preventDefault?.();
        fallbackToMainThreadFetch().then(resolve, (fallbackErr: unknown) => {
          reject(
            fallbackErr instanceof Error
              ? fallbackErr
              : new Error(String(fallbackErr)),
          );
        });
        return;
      }
      const err = new Error(
        event.message || 'Worker error while loading compressed trace data',
      );
      console.error('Failed to load compressed trace data:', err);
      reject(err);
    };
    const request: WorkerTraceRequest = {
      url: streamUrlObj.toString(),
      cacheKey,
      bypassCache,
      persistCache,
    };
    worker.postMessage(request);
  });
}
