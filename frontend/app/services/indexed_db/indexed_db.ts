/**
 * Resolves with the result of an IndexedDB request, or rejects with its error.
 */
export function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => {
      resolve(request.result);
    };
    request.onerror = () => {
      reject(request.error);
    };
  });
}

/** Resolves once an IndexedDB transaction commits, or rejects if it fails. */
export function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => {
      resolve();
    };
    // A failed request aborts its transaction, which then holds the error of
    // the request. An explicit abort leaves no error.
    transaction.onabort = () => {
      reject(
        transaction.error ??
          new DOMException('The transaction was aborted.', 'AbortError'),
      );
    };
  });
}

/** Options of an `IndexedDbConnection`. */
export interface IndexedDbConnectionOptions {
  /** Name of the database. */
  readonly name: string;
  /** Version of the schema. Bump it to run `upgrade` again. */
  readonly version: number;
  /**
   * Creates the object stores and indexes of the database, when it is created
   * or its version is bumped.
   */
  readonly upgrade: (database: IDBDatabase) => void;
  /** The IndexedDB factory, the browser's by default. */
  readonly factory?: IDBFactory;
  /**
   * How long to wait before opening the database again after it failed to
   * open, 30 s by default.
   */
  readonly retryDelayMs?: number;
  /** Returns the current time in milliseconds, `Date.now` by default. */
  readonly now?: () => number;
}

const DEFAULT_RETRY_DELAY_MS = 30_000;

/**
 * A lazily opened connection to an IndexedDB database. `get` resolves
 * undefined whenever the database cannot be used, for example because the
 * browser disables IndexedDB, so that callers can do without it.
 *
 * The connection recovers on its own. It gives up on an open that another tab
 * blocks by holding an older version of the database open, instead of
 * waiting, and retries a failed open after a delay. It reopens a connection
 * that the browser closed, or that it closed for another tab to upgrade the
 * database, on next use.
 */
export class IndexedDbConnection {
  private readonly name: string;
  private readonly version: number;
  private readonly upgrade: (database: IDBDatabase) => void;
  private readonly factory: IDBFactory | undefined;
  private readonly retryDelayMs: number;
  private readonly now: () => number;
  /** The open or opening connection, if any. */
  private connection?: Promise<IDBDatabase | undefined>;
  /** When to try opening the database again after it failed to open. */
  private retryAt = 0;
  /**
   * Whether an open that another tab blocked is still pending. Later opens
   * would queue behind it, so they wait until it settles.
   */
  private isBlockedOpenPending = false;

  constructor(options: IndexedDbConnectionOptions) {
    this.name = options.name;
    this.version = options.version;
    this.upgrade = options.upgrade;
    this.factory =
      options.factory ??
      (typeof indexedDB === 'undefined' ? undefined : indexedDB);
    this.retryDelayMs = options.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS;
    this.now = options.now ?? (() => Date.now());
  }

  /** Returns the open database, or undefined if it cannot be used. */
  get(): Promise<IDBDatabase | undefined> {
    if (this.connection) {
      return this.connection;
    }
    if (
      !this.factory ||
      this.isBlockedOpenPending ||
      this.now() < this.retryAt
    ) {
      return Promise.resolve(undefined);
    }
    const connection = this.open(this.factory);
    this.connection = connection;
    // Registered before the callbacks of callers, so that a failed connection
    // is forgotten before callers see undefined and call `get` again.
    void connection.then((database) => {
      if (database) {
        this.watch(database, connection);
      } else {
        this.forget(connection);
      }
    });
    return connection;
  }

  /** Closes the connection. The next `get` opens it again. */
  async close(): Promise<void> {
    const connection = this.connection;
    this.connection = undefined;
    (await connection)?.close();
  }

  /** Opens the database, or resolves undefined if it fails to open. */
  private async open(factory: IDBFactory): Promise<IDBDatabase | undefined> {
    try {
      return await this.sendOpenRequest(factory);
    } catch (error) {
      if (error instanceof DOMException && error.name === 'VersionError') {
        // Another tab upgraded the database past the version of this page,
        // which can never open it.
        console.warn(
          `IndexedDB database ${this.name} has a newer version. Reload the ` +
            'page to use it.',
        );
        this.retryAt = Infinity;
      } else {
        console.warn(`IndexedDB database ${this.name} is unavailable:`, error);
        this.retryAt = this.now() + this.retryDelayMs;
      }
      return undefined;
    }
  }

  /**
   * Opens the database. Rejects if the open fails, `upgrade` throws, or another
   * tab blocks the open.
   */
  private sendOpenRequest(factory: IDBFactory): Promise<IDBDatabase> {
    return new Promise<IDBDatabase>((resolve, reject) => {
      const request = factory.open(this.name, this.version);
      let isBlocked = false;
      request.onupgradeneeded = () => {
        try {
          this.upgrade(request.result);
        } catch (error) {
          // Aborting the upgrade fails the open, which then rejects with the
          // error of `upgrade` rather than an `AbortError`.
          reject(error instanceof Error ? error : new Error(String(error)));
          request.transaction?.abort();
        }
      };
      request.onsuccess = () => {
        if (isBlocked) {
          // The open was given up while it was blocked. The next `get` opens
          // the database again right away.
          this.isBlockedOpenPending = false;
          this.retryAt = 0;
          request.result.close();
        } else {
          resolve(request.result);
        }
      };
      request.onerror = () => {
        this.isBlockedOpenPending = false;
        reject(request.error);
      };
      request.onblocked = () => {
        isBlocked = true;
        this.isBlockedOpenPending = true;
        reject(new Error('The database is open in another tab.'));
      };
    });
  }

  /** Forgets the connection once it closes, so that `get` opens it again. */
  private watch(
    database: IDBDatabase,
    connection: Promise<IDBDatabase | undefined>,
  ): void {
    database.onversionchange = () => {
      // Lets another tab upgrade the database.
      database.close();
      this.forget(connection);
    };
    database.onclose = () => {
      this.forget(connection);
    };
  }

  /** Forgets `connection`, unless a newer connection replaced it. */
  private forget(connection: Promise<IDBDatabase | undefined>): void {
    if (this.connection === connection) {
      this.connection = undefined;
    }
  }
}
