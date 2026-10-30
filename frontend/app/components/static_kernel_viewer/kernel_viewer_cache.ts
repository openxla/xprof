import {type KernelViewerStore} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_viewer_store';

// How long to wait for a cached list or trace before requesting it from the
// backend instead. Reads normally take milliseconds, but wait for earlier
// writes to commit.
const READ_TIMEOUT_MS = 1000;
// The backend answers with an empty trace when it cannot read the profile,
// which could be temporary. JSON traces up to this size are checked for that.
const EMPTY_TRACE_CHECK_BYTES = 1024;

/** Downloads a trace, like `fetch`. */
export type TraceDownloader = (
  url: string,
  init?: RequestInit,
) => Promise<Response>;

/** Identifies a kernel of a profile. */
export interface KernelRef {
  /** Identifies the profile, for example by its session and host. */
  readonly profileKey: string;
  /** HLO module of the kernel, or '' if the backend reports no modules. */
  readonly module: string;
  /** Name of the kernel. */
  readonly kernel: string;
}

/** Returns the key that the cache stores the trace of a kernel under. */
export function traceKeyOf({profileKey, module, kernel}: KernelRef): string {
  // NUL cannot occur in the parts, so different kernels never share a key.
  return `${profileKey}\u0000${module}\u0000${kernel}`;
}

/** Options of `KernelViewerCache.loadList`. */
export interface LoadListOptions {
  /** Loads the backend's kernel list response. */
  readonly fetchList: () => Promise<unknown>;
  /**
   * Whether a list is worth caching and serving from the cache, for example
   * only lists with kernels.
   */
  readonly isCacheable?: (response: unknown) => boolean;
}

/**
 * The trace of a kernel. `response` is what `LoadTraceDataOptions.fetchData`
 * resolves with, and the viewer that loads it calls `save`.
 */
export interface FetchedTrace {
  readonly response: Response;
  /**
   * Caches a downloaded trace, and resolves once it is cached. The viewer
   * calls this once it has loaded the trace without error, to keep the write
   * off the loading path. Does nothing for a trace from the cache, a failed
   * download, an empty trace, or a trace saved before.
   */
  save(): Promise<void>;
}

/**
 * A trace that started loading before the viewer asked for it, see
 * `KernelViewerCache.prefetchTrace`.
 */
export interface TracePrefetch {
  /** Whether this is the trace of `kernel`. */
  isOf(kernel: KernelRef): boolean;
  /**
   * Returns the trace like `KernelViewerCache.fetchTrace`. Rejects if the
   * prefetch was canceled before the trace loaded.
   */
  take(): Promise<FetchedTrace>;
  /** Aborts the download of the trace, if any. */
  cancel(): void;
}

/** Options of a `KernelViewerCache`. */
export interface KernelViewerCacheOptions {
  /** Stores the cached lists and traces. */
  readonly store: KernelViewerStore;
  /**
   * Downloads traces, `fetch` by default. Receives the abort signal of a
   * prefetch.
   */
  readonly download?: TraceDownloader;
  /** How long to wait for a cached list or trace, 1 s by default. */
  readonly readTimeoutMs?: number;
}

/** A trace from the cache or the backend. */
interface LoadedTrace {
  readonly status: number;
  /** The trace, unless the download failed. */
  readonly data?: Blob;
  /** Whether the trace came from the cache. */
  readonly fromCache: boolean;
}

/** Throws an `AbortError` when `signal` has been aborted. */
function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException('Aborted', 'AbortError');
  }
}

/** Whether `data` is an empty blob, or a small JSON trace without events. */
async function isEmptyTrace(data: Blob): Promise<boolean> {
  if (data.size === 0) {
    return true;
  }
  if (data.size > EMPTY_TRACE_CHECK_BYTES) {
    return false;
  }
  const text = (await data.text()).trimStart();
  if (!text.startsWith('{')) {
    // A binary trace.
    return false;
  }
  try {
    const trace: unknown = JSON.parse(text);
    const events =
      typeof trace === 'object' && trace !== null
        ? (trace as Record<string, unknown>)['traceEvents']
        : undefined;
    return Array.isArray(events) && events.length === 0;
  } catch {
    return false;
  }
}

/**
 * Caches the kernel lists and the opened kernel traces of profiles, so that
 * revisiting a profile or reopening a kernel skips the backend. Waiting on the
 * cache never holds up the backend for long: a slow read is abandoned for a
 * backend request.
 *
 * Profile data is immutable once captured, so both kernel lists and traces are
 * served from the cache when present. Which lists are worth caching is up to
 * the caller, while traces that the backend returns empty are never cached.
 */
export class KernelViewerCache {
  private readonly store: KernelViewerStore;
  private readonly download: TraceDownloader;
  private readonly readTimeoutMs: number;

  constructor(options: KernelViewerCacheOptions) {
    this.store = options.store;
    this.download =
      options.download ??
      ((url: string, init?: RequestInit) => fetch(url, init));
    this.readTimeoutMs = options.readTimeoutMs ?? READ_TIMEOUT_MS;
  }

  /**
   * Returns the kernel list of a profile, from the cache or from `fetchList`.
   * Rejects when `fetchList` rejects or returns a response that is not an
   * object, such as the `null` of a failed request.
   */
  async loadList(
    profileKey: string,
    options: LoadListOptions,
  ): Promise<unknown> {
    const {fetchList, isCacheable = () => true} = options;
    const cached = await this.readWithTimeout(this.store.readList(profileKey));
    if (typeof cached === 'object' && cached !== null && isCacheable(cached)) {
      return cached;
    }
    const response = await fetchList();
    if (typeof response !== 'object' || response === null) {
      throw new Error('The backend returned no kernel list.');
    }
    if (isCacheable(response)) {
      void this.store.writeList(profileKey, response);
    }
    return response;
  }

  /**
   * Returns the trace of a kernel, from the cache or by downloading `url`.
   * Resolves with a failed response if the download fails, and rejects on
   * network errors.
   */
  async fetchTrace(kernel: KernelRef, url: string): Promise<FetchedTrace> {
    return this.toFetchedTrace(kernel, await this.loadTrace(kernel, url));
  }

  /**
   * Starts loading the trace of a kernel before the viewer asks for it, for
   * example the kernel that the URL names while the kernel list loads. The
   * caller must take or cancel the prefetch. Errors surface once the trace is
   * taken.
   */
  prefetchTrace(kernel: KernelRef, url: string): TracePrefetch {
    const controller = new AbortController();
    const trace = this.loadTrace(kernel, url, controller.signal);
    void trace.catch(() => {});
    const traceKey = traceKeyOf(kernel);
    return {
      isOf: (other: KernelRef) => traceKeyOf(other) === traceKey,
      take: async () => this.toFetchedTrace(kernel, await trace),
      cancel: () => {
        controller.abort();
      },
    };
  }

  /** Reads a trace from the cache, or downloads it. */
  private async loadTrace(
    kernel: KernelRef,
    url: string,
    signal?: AbortSignal,
  ): Promise<LoadedTrace> {
    const data = await this.readWithTimeout(
      this.store.readTrace(traceKeyOf(kernel)),
    );
    throwIfAborted(signal);
    if (data) {
      return {status: 200, data, fromCache: true};
    }
    return this.downloadTrace(url, signal);
  }

  private toFetchedTrace(
    kernel: KernelRef,
    {status, data, fromCache}: LoadedTrace,
  ): FetchedTrace {
    // Holds on to a downloaded trace only until it is saved, to spare memory.
    let unsaved = fromCache ? undefined : data;
    return {
      response: data ? new Response(data) : new Response(null, {status}),
      save: async () => {
        const pending = unsaved;
        unsaved = undefined;
        if (pending) {
          await this.saveTrace(kernel, pending);
        }
      },
    };
  }

  /** Caches a downloaded trace, unless it is empty. */
  private async saveTrace(kernel: KernelRef, data: Blob): Promise<void> {
    if (!(await isEmptyTrace(data))) {
      await this.store.writeTrace(traceKeyOf(kernel), data);
    }
  }

  /** Waits for a cache read, or gives up on it after a timeout. */
  private readWithTimeout<T>(read: Promise<T>): Promise<T | undefined> {
    return new Promise<T | undefined>((resolve) => {
      const timer = setTimeout(() => {
        resolve(undefined);
      }, this.readTimeoutMs);
      void read.then((value) => {
        clearTimeout(timer);
        resolve(value);
      });
    });
  }

  /** Downloads a trace. Rejects on network errors. */
  private async downloadTrace(
    url: string,
    signal?: AbortSignal,
  ): Promise<LoadedTrace> {
    const response = await this.download(url, {signal});
    throwIfAborted(signal);
    if (!response.ok) {
      return {status: response.status, fromCache: false};
    }
    const data = await response.blob();
    throwIfAborted(signal);
    return {status: response.status, data, fromCache: false};
  }
}
