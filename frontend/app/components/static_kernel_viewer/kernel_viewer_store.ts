import {
  IndexedDbConnection,
  requestResult,
  transactionDone,
} from 'org_xprof/frontend/app/services/indexed_db/indexed_db';

const DATABASE_NAME = 'xprof_kernel_viewer_cache';
// Bump to drop all cached data, for example when the backend responses change.
const DATABASE_VERSION = 1;
const LIST_STORE = 'lists';
// Trace blobs, apart from their metadata so that bookkeeping never reads or
// rewrites them.
const TRACE_STORE = 'traces';
const TRACE_INFO_STORE = 'trace_info';
// Each index has the name of the record property that it indexes.
const STORED_AT_INDEX = 'storedAt';

/** Cached data expires after a week, in case the backend changed since. */
export const CACHE_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;
/** Most kernel lists to keep, of the most recently cached profiles. */
export const MAX_CACHED_LISTS = 50;
const DEFAULT_MAX_CACHED_TRACE_BYTES = 512 * 1024 * 1024;
const DEFAULT_MAX_TRACE_BYTES = 128 * 1024 * 1024;

// Records as stored in IndexedDB. `declare` keeps the compiler from renaming
// their properties, which the key paths and indexes refer to by name.

declare interface ListRecord {
  readonly key: string;
  readonly response: unknown;
  readonly storedAt: number;
}

declare interface TraceRecord {
  readonly key: string;
  readonly data: Blob;
}

declare interface TraceInfoRecord {
  readonly key: string;
  readonly size: number;
  readonly storedAt: number;
  readonly lastUsed: number;
}

/**
 * Stores the kernel lists and kernel traces of profiles. Every method treats
 * a failing storage as empty, so that the cache only ever speeds up loading.
 */
export interface KernelViewerStore {
  /** Returns the kernel list response of a profile, if stored. */
  readList(profileKey: string): Promise<unknown>;
  /** Stores the kernel list response of a profile. */
  writeList(profileKey: string, response: unknown): Promise<void>;
  /** Returns the trace of a kernel, if stored, and marks it as used. */
  readTrace(traceKey: string): Promise<Blob | undefined>;
  /** Stores the trace of a kernel. */
  writeTrace(traceKey: string, data: Blob): Promise<void>;
}

/** Options of an `IndexedDbKernelViewerStore`, for tests. */
export interface IndexedDbKernelViewerStoreOptions {
  /** Name of the database, `xprof_kernel_viewer_cache` by default. */
  readonly databaseName?: string;
  /** The IndexedDB factory, the browser's by default. */
  readonly factory?: IDBFactory;
  /** Returns the current time in milliseconds, `Date.now` by default. */
  readonly now?: () => number;
  /**
   * Size of the largest trace to store, 128 MiB by default. Traces larger than
   * `maxCachedTraceBytes` are not stored either.
   */
  readonly maxTraceBytes?: number;
  /** Most bytes of traces to keep, 512 MiB by default. */
  readonly maxCachedTraceBytes?: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isListRecord(value: unknown): value is ListRecord {
  if (!isRecord(value)) {
    return false;
  }
  const record = value as Partial<ListRecord>;
  return typeof record.key === 'string' && typeof record.storedAt === 'number';
}

function isTraceRecord(value: unknown): value is TraceRecord {
  return (
    isRecord(value) && (value as Partial<TraceRecord>).data instanceof Blob
  );
}

function isTraceInfoRecord(value: unknown): value is TraceInfoRecord {
  if (!isRecord(value)) {
    return false;
  }
  const record = value as Partial<TraceInfoRecord>;
  return (
    typeof record.key === 'string' &&
    typeof record.size === 'number' &&
    typeof record.storedAt === 'number' &&
    typeof record.lastUsed === 'number'
  );
}

function isQuotaExceeded(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'QuotaExceededError';
}

/**
 * Stores kernel lists and kernel traces in IndexedDB.
 *
 * IndexedDB is used rather than the Cache API (`window.caches`): the Cache API
 * exists only in secure contexts, so it is missing when XProf is served over
 * plain HTTP from a remote host, and it has no indexes for evicting by last
 * use or capping the total size.
 *
 * Entries expire after `CACHE_MAX_AGE_MS`, and at most `MAX_CACHED_LISTS`
 * lists are kept. Traces are capped in size, both one by one and in total,
 * and the least recently used traces go first.
 */
export class IndexedDbKernelViewerStore implements KernelViewerStore {
  private readonly connection: IndexedDbConnection;
  private readonly clock: () => number;
  private readonly maxTraceBytes: number;
  private readonly maxCachedTraceBytes: number;

  constructor(options: IndexedDbKernelViewerStoreOptions = {}) {
    this.clock = options.now ?? (() => Date.now());
    this.connection = new IndexedDbConnection({
      name: options.databaseName ?? DATABASE_NAME,
      version: DATABASE_VERSION,
      upgrade: (database) => {
        for (const name of Array.from(database.objectStoreNames)) {
          database.deleteObjectStore(name);
        }
        database
          .createObjectStore(LIST_STORE, {keyPath: 'key'})
          .createIndex(STORED_AT_INDEX, STORED_AT_INDEX);
        database.createObjectStore(TRACE_STORE, {keyPath: 'key'});
        database.createObjectStore(TRACE_INFO_STORE, {keyPath: 'key'});
      },
      factory: options.factory,
      now: this.clock,
    });
    this.maxTraceBytes = options.maxTraceBytes ?? DEFAULT_MAX_TRACE_BYTES;
    this.maxCachedTraceBytes =
      options.maxCachedTraceBytes ?? DEFAULT_MAX_CACHED_TRACE_BYTES;
  }

  /** Closes the database, for example at the end of a test. */
  close(): Promise<void> {
    return this.connection.close();
  }

  async readList(profileKey: string): Promise<unknown> {
    try {
      const database = await this.connection.get();
      if (!database) {
        return undefined;
      }
      const record: unknown = await requestResult(
        database
          .transaction(LIST_STORE, 'readonly')
          .objectStore(LIST_STORE)
          .get(profileKey),
      );
      if (!isListRecord(record) || this.isExpired(record.storedAt)) {
        return undefined;
      }
      return record.response;
    } catch (error) {
      console.warn('Failed to read the cached kernel list:', error);
      return undefined;
    }
  }

  async writeList(profileKey: string, response: unknown): Promise<void> {
    try {
      const database = await this.connection.get();
      if (!database) {
        return;
      }
      const now = this.clock();
      const transaction = database.transaction(LIST_STORE, 'readwrite');
      const store = transaction.objectStore(LIST_STORE);
      const record: ListRecord = {key: profileKey, response, storedAt: now};
      store.put(record);
      // Keys come sorted by storage time, oldest first, without loading the
      // lists. Expired lists come first, so removing the oldest keys drops both
      // the expired lists and the lists beyond `MAX_CACHED_LISTS`. Lists stored
      // in the future, after a clock change, stay until they are replaced.
      const index = store.index(STORED_AT_INDEX);
      const expiredCount = await requestResult(
        index.count(IDBKeyRange.upperBound(now - CACHE_MAX_AGE_MS)),
      );
      const keys = await requestResult(index.getAllKeys());
      const evictedCount = Math.max(
        expiredCount,
        keys.length - MAX_CACHED_LISTS,
      );
      for (const key of keys.slice(0, evictedCount)) {
        store.delete(key);
      }
      await transactionDone(transaction);
    } catch (error) {
      console.warn('Failed to cache the kernel list:', error);
    }
  }

  async readTrace(traceKey: string): Promise<Blob | undefined> {
    try {
      const database = await this.connection.get();
      if (!database) {
        return undefined;
      }
      // Reads the trace and its metadata in one transaction, so that they
      // match.
      const transaction = database.transaction(
        [TRACE_STORE, TRACE_INFO_STORE],
        'readonly',
      );
      const [info, trace]: unknown[] = await Promise.all([
        requestResult(transaction.objectStore(TRACE_INFO_STORE).get(traceKey)),
        requestResult(transaction.objectStore(TRACE_STORE).get(traceKey)),
      ]);
      if (
        !isTraceInfoRecord(info) ||
        !isTraceRecord(trace) ||
        this.isExpired(info.storedAt)
      ) {
        return undefined;
      }
      void this.markUsed(database, info);
      return trace.data;
    } catch (error) {
      console.warn('Failed to read the cached kernel trace:', error);
      return undefined;
    }
  }

  async writeTrace(traceKey: string, data: Blob): Promise<void> {
    if (data.size > Math.min(this.maxTraceBytes, this.maxCachedTraceBytes)) {
      return;
    }
    try {
      const database = await this.connection.get();
      if (!database) {
        return;
      }
      // Makes room first, in its own transaction, so that a write that fails
      // for lack of space still frees it.
      const keptBytes = await this.evictTraces(
        database,
        this.maxCachedTraceBytes - data.size,
        traceKey,
      );
      try {
        await this.putTrace(database, traceKey, data);
      } catch (error) {
        if (!isQuotaExceeded(error)) {
          throw error;
        }
        // The browser grants less space than the budget. Halves the cache,
        // and tries once more.
        await this.evictTraces(database, Math.floor(keptBytes / 2), traceKey);
        await this.putTrace(database, traceKey, data);
      }
    } catch (error) {
      console.warn('Failed to cache the kernel trace:', error);
    }
  }

  private async putTrace(
    database: IDBDatabase,
    traceKey: string,
    data: Blob,
  ): Promise<void> {
    const transaction = database.transaction(
      [TRACE_STORE, TRACE_INFO_STORE],
      'readwrite',
    );
    const now = this.clock();
    const trace: TraceRecord = {key: traceKey, data};
    const info: TraceInfoRecord = {
      key: traceKey,
      size: data.size,
      storedAt: now,
      lastUsed: now,
    };
    transaction.objectStore(TRACE_STORE).put(trace);
    transaction.objectStore(TRACE_INFO_STORE).put(info);
    await transactionDone(transaction);
  }

  /**
   * Records when a stored trace was last used, for eviction. Leaves the trace
   * alone if it was rewritten or removed since `info` was read.
   */
  private async markUsed(
    database: IDBDatabase,
    info: TraceInfoRecord,
  ): Promise<void> {
    try {
      const transaction = database.transaction(TRACE_INFO_STORE, 'readwrite');
      const infos = transaction.objectStore(TRACE_INFO_STORE);
      const current: unknown = await requestResult(infos.get(info.key));
      if (isTraceInfoRecord(current) && current.storedAt === info.storedAt) {
        const used: TraceInfoRecord = {...current, lastUsed: this.clock()};
        infos.put(used);
      }
      await transactionDone(transaction);
    } catch (error) {
      console.warn('Failed to update the cached kernel trace:', error);
    }
  }

  /**
   * Removes expired traces, and the least recently used ones beyond `budget`
   * bytes. Leaves out `replacedKey`, which is about to be rewritten. Returns
   * the size of the traces that remain.
   */
  private async evictTraces(
    database: IDBDatabase,
    budget: number,
    replacedKey: string,
  ): Promise<number> {
    const transaction = database.transaction(
      [TRACE_STORE, TRACE_INFO_STORE],
      'readwrite',
    );
    const traces = transaction.objectStore(TRACE_STORE);
    const infos = transaction.objectStore(TRACE_INFO_STORE);
    const records: unknown[] = await requestResult(infos.getAll());
    const live: TraceInfoRecord[] = [];
    for (const record of records) {
      if (!isTraceInfoRecord(record) || record.key === replacedKey) {
        continue;
      }
      if (this.isExpired(record.storedAt)) {
        traces.delete(record.key);
        infos.delete(record.key);
      } else {
        live.push(record);
      }
    }
    live.sort((a, b) => b.lastUsed - a.lastUsed);
    let total = 0;
    for (const record of live) {
      if (total + record.size > budget) {
        traces.delete(record.key);
        infos.delete(record.key);
      } else {
        total += record.size;
      }
    }
    await transactionDone(transaction);
    return total;
  }

  /** Whether data stored at `storedAt` expired, or comes from the future. */
  private isExpired(storedAt: number): boolean {
    const age = this.clock() - storedAt;
    return age < 0 || age >= CACHE_MAX_AGE_MS;
  }
}
