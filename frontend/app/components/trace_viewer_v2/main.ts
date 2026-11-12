
import {getDefaultFeatureFlag} from './feature_flags';
import {
  getFetchErrorMessage,
  isAbortError,
  isClientOpfsBypassRequested,
  loadCompressedTraceDataInternal,
  shouldBypassOpfsCache,
  streamCompressedTraceFrames,
  type TraceDataFetcher,
  type TraceStreamWorkerFactory,
} from './trace_stream_worker';

export type {TraceDataFetcher, TraceStreamWorkerFactory};

/**
 * The over-fetching factor for trace events.
 *
 * We request ZOOM_RATIO times more events (resolution bins) than the current
 * viewport width requires. This ensures that we have enough data to allow the
 * user to zoom in up to this factor without losing detail (i.e., having to
 * re-fetch data because the resolution became too coarse).
 */
export const ZOOM_RATIO = 8;

/**
 * The over-fetching factor for the initial fetch.
 *
 * We request FETCH_RATIO times more data than the current viewport width
 * requires for the initial fetch.
 */
export const FETCH_RATIO = 3.0;

/**
 * The minimum duration of a fetch request in milliseconds (= 1 microsecond).
 *
 * The initial fetch for a URL time range must cover the same window that the
 * WASM timeline records as fetched; otherwise the timeline assumes it already
 * has data that was never requested and does not fetch it on pan or zoom.
 */
export const MIN_FETCH_DURATION_MS = 0.001;

/**
 * The width of the left-side label column in the trace viewer in pixels.
 *
 * This corresponds to the `label_width_` in `timeline.h`.
 */
export const HEADING_WIDTH = 250;

/**
 * Minimum event width in logical pixels used for calculating resolution.
 *
 * Events smaller than this threshold are generally not visible and difficult to
 * interact with. The backend uses this to downsample events to improve
 * loading performance.
 */
export const MIN_EVENT_WIDTH = 2;

/**
 * URL parameters corresponding to `TraceOptions`.
 *
 * See third_party/xprof/convert/trace_viewer/trace_options.h
 */
export const TRACE_OPTIONS = {
  SELECTED_GROUP_IDS: 'selected_group_ids',
  MPMD_PIPELINE_VIEW_PARAM: 'mpmd_pipeline_view',
} as const;

/**
 * URL parameters corresponding to `TraceViewOption`.
 *
 * See third_party/xprof/convert/xplane_to_tools_data.cc
 */
export const TRACE_VIEW_OPTION = {
  RESOLUTION: 'resolution',
  START_TIME_MS: 'start_time_ms',
  END_TIME_MS: 'end_time_ms',
} as const;

/**
 * URL parameters for initial view start.
 *
 * This is used to set the initial view when the trace viewer is first loaded.
 */
export const VIEW_START = 'view_start';

/**
 * URL parameters for initial view end.
 *
 * This is used to set the initial view when the trace viewer is first loaded.
 */
export const VIEW_END = 'view_end';

/**
 * The name of the loading status update custom event, dispatched from WASM in
 * Trace Viewer v2.
 */
export const LOADING_STATUS_UPDATE_EVENT_NAME = 'loadingstatusupdate';

/**
 * Event dispatched when details (like full_dma) are returned by the backend.
 */
export const DETAILS_RECEIVED_EVENT_NAME = 'details_received';

/** Represents trace details toggles. */
/** Supported trace detail keys. */
export type TraceDetailKey = 'full_dma';

/** Represents trace details toggles. */
export type TraceDetails = Map<TraceDetailKey, boolean>;

/** Detail of the event dispatched when trace details are received. */
export declare interface DetailsReceivedEventDetail {
  details: TraceDetails;
}

/** Type guard for DetailsReceivedEvent. */
export function isDetailsReceivedEvent(
  event: Event,
): event is CustomEvent<DetailsReceivedEventDetail> {
  return (
    event instanceof CustomEvent &&
    !!event.detail &&
    event.detail.details instanceof Map
  );
}

/**
 * The name of the request data custom event, dispatched from the UI when the
 * user requests new data (e.g. by zooming or panning).
 */
export const FETCH_DATA_EVENT_NAME = 'fetch_data';

/**
 * The name of the search event, dispatched from WASM when a search is
 * requested.
 */
export const SEARCH_EVENTS_EVENT_NAME = 'search_events';

/**
 * The loading status of the trace viewer, used to update the loading status
 * indicator in the UI.
 */
export enum TraceViewerV2LoadingStatus {
  IDLE = 'Idle',
  INITIALIZING = 'Initializing',
  LOADING_DATA = 'Loading data',
  PROCESSING_DATA = 'Processing data',
  ERROR = 'Error',
}

declare function loadWasmTraceViewerModule(
  options?: object,
): Promise<TraceViewerV2Module>;

declare global {
  interface Window {
    wasmMemoryBytes: number;
    getFeatureFlag?: (name: string) => boolean;
  }
}

  export declare interface TraceViewerV2Module extends EmscriptenModule {
  HEAPU8: Uint8Array;
  _malloc(size: number): number;
  _free(ptr: number): void;
  getFeatureFlag?(name: string): boolean;
  SetPalette(paletteName: string): void;
  SetPanningSpeed?(speed: number): void;
  SetZoomSpeed?(speed: number): void;
  SetMouseWheelZoomSpeed?(speed: number): void;
  /**
   * Enables/disables `fetch_data` requests for time-range refetching. Turn off
   * when the backend always returns the full dataset.
   */
  SetIncrementalFetchEnabled?(enabled: boolean): void;
  /**
   * Clears all per-dataset state (data, viewport, selection, search, track
   * metadata). Call before `loadTraceData` when switching datasets.
   */
  ResetDataset?(): void;
  SetCustomTraceColors?(colors: number[]): void;
  RequestRedraw?(): void;
  SetPlaybackState?(
    isPlaying: boolean,
    currentTime: number,
    playSpeed: number,
  ): void;
  GetPresetPalettes?(): Array<{
    name: string;
    previewColors: string[];
    backgroundColor?: string;
  }>;
  /**
   * Shows or hides the built-in tooltip of hovered events. On by default;
   * hosts that render their own tooltip from the `eventhovered` event turn it
   * off. Hover highlighting, hover events and selection are unaffected.
   */
  SetEventTooltipEnabled?(enabled: boolean): void;
  /**
   * Whether selecting an event also draws a temporary time range curtain
   * spanning it. Off by default; the Kernel Viewer turns it on.
   */
  SetSelectedEventTimeRangeEnabled?(enabled: boolean): void;
  /** Enables or disables reordering tracks by their drag handle. */
  SetTrackReorderingEnabled?(enabled: boolean): void;
  /**
   * Sets the `localStorage` key used for saving and restoring the track order
   * (`xprof_trace_viewer_track_order` by default), and immediately restores
   * the order saved under the new key.
   */
  SetTrackOrderStorageKey?(key: string): void;
  /**
   * Expands every process track by default instead of only the first process.
   * Off by default; the Kernel Viewer turns it on so `Memory` and `Utilization`
   * start expanded alongside the schedule.
   */
  SetExpandAllProcessesByDefault?(enabled: boolean): void;
  /**
   * Configures thread-level tracks that start collapsed to a single line when
   * they span multiple levels. Empty by default.
   */
  SetDefaultCollapsedTracks?(trackNames: readonly string[]): void;
  // Sets the visible time range of the timeline. Both arguments are in
  // microseconds. Registered as a module-level Emscripten function (see
  // `SetVisibleRange` in application.cc). Used by the Live Trace Viewer to
  // auto-follow the newest sliding window as streaming data arrives.
  setVisibleRange(startUs: number, endUs: number): void;
  /**
   * Passes the HLO operand and consumer names of the selected event, so that
   * the timeline can draw the HLO dependency arrows (behind the
   * `enable_hlo_dependency_arrows` feature flag). Registered as a module-level
   * Emscripten function (see `SetSelectedEventDependencies` in
   * application.cc). Call once the names of the selected op are known.
   */
  SetSelectedEventDependencies?(
    operandNames: string[],
    consumerNames: string[],
  ): void;
  canvas: HTMLCanvasElement;
  callMain(args: string[]): void;
  preinitializedWebGPUDevice: GPUDevice | null;
  processTraceEvents(
    data: TraceData,
    timeRangeFromUrl?: [number, number],
  ): void;
  /**
   * Passes compressed protobuf trace events from a memory buffer in the WASM heap.
   * @param dataPtr A pointer to the memory address in the WASM heap where
   *     the compressed trace data is stored.
   * @param dataSize The size of the compressed trace data in bytes.
   * @param timeRangeFromUrl Optional initial visible time range [start, end]
   *     in milliseconds.
   */
  processCompressedTraceEvents(
    dataPtr: number,
    dataSize: number,
    timeRangeFromUrl?: [number, number],
  ): void;
  beginIncrementalTraceEvents?(timeRangeFromUrl?: [number, number]): void;
  hasCachedTraceEventsIncremental?(): boolean;
  shouldEnqueueCompressedFrameIncremental?(): boolean;
  appendCompressedTraceEventsIncremental?(
    dataPtr: number,
    dataSize: number,
  ): void;
  stepIncrementalTraceEvents?(maxEventsPerStep: number): boolean;
  finishIncrementalTraceEvents?(): boolean | void;
  cancelIncrementalTraceEvents?(): void;
  restoreFullTraceOverviewIfAvailable?(): boolean;
  /**
   * Passes Perfetto trace events from a memory buffer in the WASM heap.
   * @param dataPtr A pointer to the memory address in the WASM heap where
   *     the Perfetto trace data is stored.
   * @param dataSize The size of the Perfetto trace data in bytes.
   * @param timeRangeFromUrl Optional initial visible time range [start, end]
   *     in milliseconds.
   * @param normalizeTimestamps If true, timestamps will be normalized to
   *     start from 0 if they seem to be absolute.
   */
  processPerfettoTraceEvents(
    dataPtr: number,
    dataSize: number,
    timeRangeFromUrl: [number, number] | undefined,
    normalizeTimestamps: boolean,
  ): void;
  getAllFlowCategories(): Array<{id: number; name: string}>;
  setCompressedSearchResultsInWasm(data: Uint8Array): void;
  setSearchResultsInWasm(data: TraceData): void;
  /**
   * Loads and shows the trace at a URL. A call with the URL that is already
   * loading returns that load, and a call with another URL replaces it.
   */
  loadTraceData?(url: string, options?: LoadTraceDataOptions): Promise<void>;
  loadSearchResults?(url: string): Promise<void>;
  StringVector: {
    size(): number;
    get(index: number): string;
    toArray(): string[];
  };
  IntVector: {size(): number; get(index: number): number};
  application: {
    instance(): {
      shutdown(): void;
      dataProvider(): {
        getFlowCategories(): TraceViewerV2Module['IntVector'];
        getProcessMappings(): Record<number, string>;
        getProcessNames(): Record<number, string>;
      };
      getCurrentSearchResultIndex(): number;
      getSearchResultsCount(): number;
      navigateToNextSearchResult(): void;
      navigateToPrevSearchResult(): void;
      /** Reveals the flow-connected event in the timeline by its index. */
      revealEvent?(eventIndex: number): void;
      revealEvent?(
        eventIndex: number,
        name: string,
        startUs: number,
        durationUs: number,
      ): void;
      resize(dpr: number, width: number, height: number): void;
      /**
       * Re-applies the track order saved in `localStorage`, which all sessions
       * share, so that a newly loaded trace uses the latest layout.
       */
      restoreSavedTrackOrder?(): void;
      setSearchQuery(query: string): void;
      selectEvent(eventIndex: number): void;
      setMouseMode(mode: number): void;
      setVisibleRange?(startUs: number, endUs: number): void;
      setVisibleFlowCategory(categoryId: number): void;
      setVisibleFlowCategories(categoryIds: number[]): void;
      scheduleForcedRedraw(): void;
      zoomIn(): void;
      zoomOut(): void;
      /** Animates the visible range to the full time range of the trace. */
      resetView?(): void;
      /** Zooms to the selected event. Does nothing if none is selected. */
      zoomToSelection?(): void;
    };
  };
}

/**
 * Options for `TraceViewerV2Module.loadTraceData`.
 */
export interface LoadTraceDataOptions {
  /**
   * Downloads the trace in place of `fetch`, for example to serve it from a
   * cache. Receives the URL that the viewer would fetch, including the
   * parameters that the viewer adds. To report an error, rejects or resolves
   * with a failed response; to cancel the load without reporting an error,
   * rejects with an `AbortError` `DOMException`, like an aborted `fetch`.
   * Applies to this load only: time-range refetches and search results still
   * use `fetch`. A call with a URL that is already loading joins that load and
   * ignores its options.
   */
  fetchData?: TraceDataFetcher;
  /**
   * Optional factory used to construct the dedicated trace-streaming Web
   * Worker (for example, to inject an isolated worker test harness). Defaults
   * to creating a standard browser `Worker` from the inline worker Blob URL.
   */
  createStreamWorker?: TraceStreamWorkerFactory;
}

/**
 * Interface for the trace data loaded by the trace viewer.
 */
export declare interface TraceData {
  traceEvents: Array<{[key: string]: unknown}>;
  fullTimespan?: [number, number];
  details?: unknown;
  [key: string]: unknown;
}

/**
 * Type guard to check if an object conforms to the TraceData interface.
 * @param data The object to check.
 * @return True if the object is a TraceData object.
 */
export function isTraceData(data: unknown): data is TraceData {
  return (
    typeof data === 'object' &&
    data !== null &&
    Object.prototype.hasOwnProperty.call(data, 'traceEvents') &&
    Array.isArray((data as Record<string, unknown>)['traceEvents'])
  );
}

/**
 * Dispatches a DETAILS_RECEIVED_EVENT if the trace data contains details.
 */
function maybeDispatchDetailsReceivedEvent(jsonData: TraceData) {
  const rawDetails = jsonData['details'];
  if (!rawDetails) return;

  const details: TraceDetails = new Map();
  if (Array.isArray(rawDetails)) {
    for (const d of rawDetails) {
      if (d && typeof d === 'object') {
        const item = d as Record<string, unknown>;
        if (item['name'] === 'full_dma' && typeof item['value'] === 'boolean') {
          details.set('full_dma', item['value']);
        }
      }
    }
  } else if (typeof rawDetails === 'object' && rawDetails !== null) {
    const rawDetailsObj = rawDetails as Record<string, unknown>;
    const fullDmaValue = rawDetailsObj['full_dma'];
    if (typeof fullDmaValue === 'boolean') {
      details.set('full_dma', fullDmaValue);
    }
  }

  if (details.size > 0) {
    window.dispatchEvent(
      new CustomEvent(DETAILS_RECEIVED_EVENT_NAME, {
        detail: {details},
      }),
    );
  }
}

// Global state to track active WASM module and event listeners for cleanup.
let activeWasmModule: TraceViewerV2Module | null = null;
let activeWasmIncrementalSessionId = 0;
let activeLoadAbortController: AbortController | null = null;
let activeRefetchAbortController: AbortController | null = null;

/**
 * Interface for tracking event listeners registered on the window.
 * Each instance stores the event `type` and the `listener` function,
 * allowing them to be properly removed when the trace viewer is shut down.
 */
interface RegisteredListener {
  type: string;
  listener: EventListener;
}

const MAX_CACHED_COMPRESSED_BYTES = 512 * 1024 * 1024;
let cachedFullTraceUrl: string | null = null;
let cachedFullTraceFrames: Uint8Array[] | null = null;

function clearCachedFullTrace(): void {
  cachedFullTraceUrl = null;
  cachedFullTraceFrames = null;
}

function getCanonicalTraceCacheKey(urlObj: URL): string {
  const copy = new URL(urlObj.toString());
  copy.searchParams.delete(TRACE_VIEW_OPTION.RESOLUTION);
  copy.searchParams.delete(TRACE_VIEW_OPTION.START_TIME_MS);
  copy.searchParams.delete(TRACE_VIEW_OPTION.END_TIME_MS);
  copy.searchParams.delete(VIEW_START);
  copy.searchParams.delete(VIEW_END);
  copy.searchParams.delete('stream');
  copy.searchParams.delete('use_saved_result');
  copy.searchParams.delete('nocache');
  copy.searchParams.sort();
  return copy.toString();
}

const registeredEventListeners: RegisteredListener[] = [];

function registerWindowListener(type: string, listener: EventListener) {
  window.addEventListener(type, listener);
  registeredEventListeners.push({type, listener});
}

/**
 * Shuts down the active Trace Viewer v2 WASM application and cleans up
 * resources, including event listeners and WASM memory.
 */
export function shutdownTraceViewerV2() {
  ++activeWasmIncrementalSessionId;
  activeLoadAbortController?.abort();
  activeLoadAbortController = null;
  activeRefetchAbortController?.abort();
  activeRefetchAbortController = null;
  clearCachedFullTrace();
  if (activeWasmModule) {
    try {
      activeWasmModule.application.instance().shutdown();
    } catch (e) {
      console.error('Error during WASM shutdown:', e);
    }
    activeWasmModule = null;
  }

  for (const {type, listener} of registeredEventListeners) {
    window.removeEventListener(type, listener);
  }
  registeredEventListeners.length = 0;
}

/**
 * Returns the currently active Trace Viewer v2 WASM module, or null if
 * the module has not been initialized or has been shut down.
 */
export function getActiveWasmModule(): TraceViewerV2Module | null {
  return activeWasmModule;
}

/**
 * Extension of GPUDevice that includes the lost promise for device disconnection handling.
 */
export interface GPUDeviceWithLost extends GPUDevice {
  readonly lost: Promise<GPUDeviceLostInfo>;
}

/**
 * Monitors a WebGPU device for disconnection and dispatches an error status event when lost.
 */
export function monitorDeviceLost(device: GPUDeviceWithLost): void {
  void device.lost
    .then((info) => {
      const msg = `WebGPU device was lost: ${
        info?.message ?? 'unknown'
      }. Please reload the page.`;
      dispatchErrorStatus(msg, new Error(msg));
    })
    .catch(() => {});
}

async function getWebGpuDevice(): Promise<GPUDevice> {
  const gpu = navigator.gpu;
  if (!gpu) {
    throw new Error(
      'WebGPU is not supported on this browser. Please use a WebGPU-compatible browser (e.g. Chrome) and ensure hardware acceleration is enabled.',
    );
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    throw new Error(
      'WebGPU cannot be initialized: graphics adapter not found. Please ensure hardware acceleration is enabled in your browser settings (e.g. chrome://settings/system).',
    );
  }
  const device = await adapter.requestDevice();
  if (!device) {
    throw new Error(
      'WebGPU cannot be initialized: failed to get WebGPU device. Please ensure hardware acceleration is enabled.',
    );
  }
  monitorDeviceLost(device as GPUDeviceWithLost);
  return device;
}

function configureCanvas(canvas: HTMLCanvasElement, device: GPUDevice) {
  const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
  if (!context) {
    throw new Error('Context not found for canvas.');
  }
  context.configure({
    device,
    format: navigator.gpu.getPreferredCanvasFormat(),
  });
}

/**
 * Selector for the `<script>` element that loads the Emscripten glue code of
 * the WASM module.
 */
const WASM_GLUE_SCRIPT_SELECTOR = 'script[src*="trace_viewer_v2.js"]';

/**
 * Returns the query string (e.g. `?v=123`) of the WASM glue script URL, or an
 * empty string if the script is not found or its URL has no query.
 */
function getWasmGlueScriptQuery(): string {
  const script = document.querySelector<HTMLScriptElement>(
    WASM_GLUE_SCRIPT_SELECTOR,
  );
  return script ? new URL(script.src, document.baseURI).search : '';
}

async function loadAndStartWasm(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
): Promise<TraceViewerV2Module> {
  // The glue code and the WASM binary must come from the same build, otherwise
  // the initialization fails with `ASM_CONSTS[code] is not a function`. Fetch
  // the binary with the same query (e.g. a cache-busting version) as the glue
  // script, so that browser and intermediary caches serve a matching pair.
  const glueScriptQuery = getWasmGlueScriptQuery();
  const moduleConfig = {
    canvas,
    print: console.log,
    printErr: console.error,
    setStatus: console.debug,
    noInitialRun: true,
    locateFile: (path: string, scriptDirectory: string) =>
      `${scriptDirectory}${path}${glueScriptQuery}`,
  };

  performance.mark('wasmLoadStart');

  const traceviewerModule = await loadWasmTraceViewerModule(moduleConfig);

  performance.mark('wasmLoadEnd');
  performance.measure('wasmModuleLoadTime', 'wasmLoadStart', 'wasmLoadEnd');

  traceviewerModule.preinitializedWebGPUDevice = device;

  performance.mark('appInitStart');

  traceviewerModule.callMain([]);

  performance.mark('appInitEnd');
  performance.measure('appInitializationTime', 'appInitStart', 'appInitEnd');

  return traceviewerModule;
}

async function ensureWasmModuleIsLoaded(): Promise<void> {
  // tslint:disable-next-line:no-any
  if (typeof (window as any).loadWasmTraceViewerModule !== 'undefined') {
    return;
  }
  return new Promise((resolve, reject) => {
    const existingScript = document.querySelector(WASM_GLUE_SCRIPT_SELECTOR);
    if (existingScript) {
      existingScript.addEventListener('load', () => {
        resolve();
      });
      existingScript.addEventListener('error', () => {
        reject(new Error('Failed to load WASM module.'));
      });
      return;
    }
    const script = document.createElement('script');
    script.src = 'trace_viewer_v2.js';
    script.onload = () => {
      resolve();
    };
    script.onerror = () => {
      reject(new Error('Failed to load WASM module.'));
    };
    document.body.appendChild(script);
  });
}

async function initGpuAndStartWasmApp(): Promise<TraceViewerV2Module> {
  await ensureWasmModuleIsLoaded();
  const canvas = document.querySelector('#canvas') as HTMLCanvasElement;
  if (!canvas) {
    throw new Error('Could not find canvas element with id="canvas"');
  }
  const device = await getWebGpuDevice();
  configureCanvas(canvas, device);
  return loadAndStartWasm(canvas, device);
}

/**
 * Sets up drag-and-drop and file input handlers for uploading trace files.
 *
 * @param traceviewerModule The initialized Trace Viewer v2 WASM module.
 * @param onFileProcessed Optional callback to execute when a file is
 *     successfully processed.
 */
function setupFileInputHandler(
  traceviewerModule: TraceViewerV2Module,
  onFileProcessed?: () => void,
) {
  const fileInput = document.getElementById('fileInput') as HTMLInputElement;
  if (fileInput) {
    fileInput.addEventListener('change', async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (!file) {
        return;
      }
      await processUploadedFile(file, traceviewerModule, onFileProcessed);
    });
  }

  function isDragEvent(event: Event): event is DragEvent {
    return event instanceof DragEvent;
  }

  // Set up drag-and-drop on the window/document body or canvas.
  const handleDragOver = (event: Event) => {
    if (!isDragEvent(event)) {
      return;
    }
    event.preventDefault();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'copy';
    }
  };

  const handleDrop = async (event: Event) => {
    if (!isDragEvent(event)) {
      return;
    }
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (!file) {
      return;
    }
    await processUploadedFile(file, traceviewerModule, onFileProcessed);
  };

  registerWindowListener('dragover', handleDragOver);
  registerWindowListener('drop', handleDrop);
}

/**
 * Dispatches an ERROR loading status event to the window and logs the message.
 */
function dispatchErrorStatus(msg: string, error: unknown) {
  console.error(msg, error);

  window.dispatchEvent(
    new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
      detail: {
        status: TraceViewerV2LoadingStatus.ERROR,
        message: msg,
      },
    }),
  );
}

function hasGzipMagicNumber(data: Uint8Array): boolean {
  return data.length > 2 && data[0] === 0x1f && data[1] === 0x8b;
}

// DecompressionStream is a standard Web API, but its types might be missing
// in some TypeScript environments (e.g. presubmit). We declare it here to
// fix compilation errors without using 'any'.
declare class DecompressionStream {
  constructor(format: 'gzip' | 'deflate');
  readonly readable: ReadableStream;
  readonly writable: WritableStream;
}

async function decompressGzip(data: Uint8Array): Promise<Uint8Array | null> {
  try {
    const ds = new DecompressionStream('gzip');
    const response = new Response(data.buffer as ArrayBuffer);
    if (!response.body) {
      throw new Error('Failed to create response body for decompression');
    }
    const decompressedStream = response.body.pipeThrough(ds);
    const decompressedArrayBuffer = await new Response(
      decompressedStream,
    ).arrayBuffer();
    return new Uint8Array(decompressedArrayBuffer as ArrayBuffer);
  } catch (error) {
    dispatchErrorStatus('Failed to decompress gzipped file.', error);
    return null;
  }
}

function isPerfettoTrace(data: Uint8Array, fileName: string): boolean {
  return (
    fileName.endsWith('.pftrace') ||
    fileName.endsWith('.perfetto-trace') ||
    !isJsonContent(data, fileName)
  );
}

function processPerfettoTrace(
  data: Uint8Array,
  traceviewerModule: TraceViewerV2Module,
) {
  let dataPtr: number | undefined;
  try {
    dataPtr = traceviewerModule._malloc(data.length);
    if (!dataPtr) {
      throw new Error('Failed to allocate WASM memory buffer');
    }
    traceviewerModule.HEAPU8.set(data, dataPtr);

    traceviewerModule.processPerfettoTraceEvents(
      dataPtr,
      data.length,
      undefined,
      true,
    );
  } catch (error) {
    dispatchErrorStatus(
      error instanceof Error ? error.message : String(error),
      error,
    );
  } finally {
    if (dataPtr !== undefined && dataPtr !== 0) {
      traceviewerModule._free(dataPtr);
    }
  }
}

function isJsonContent(data: Uint8Array, fileName: string): boolean {
  if (fileName.endsWith('.json')) {
    return true;
  }
  for (let i = 0; i < Math.min(data.length, 100); i++) {
    const char = String.fromCharCode(data[i]);
    if (char === '{' || char === '[') {
      return true;
    }
    if (!/\s/.test(char)) {
      break;
    }
  }
  return false;
}

function processJsonTrace(
  data: Uint8Array,
  traceviewerModule: TraceViewerV2Module,
) {
  try {
    const decoder = new TextDecoder('utf-8');
    const fileContent = decoder.decode(data);
    const jsonData = JSON.parse(fileContent) as unknown;

    if (!isTraceData(jsonData)) {
      throw new Error('File does not contain valid trace events.');
    }

    traceviewerModule.processTraceEvents(jsonData, undefined);
  } catch (error) {
    dispatchErrorStatus(
      error instanceof Error ? error.message : String(error),
      error,
    );
    throw error;
  }
}

/**
 * Processes an uploaded file containing trace data.
 *
 * If the file is a Perfetto trace (.pftrace or .perfetto-trace), it is
 * processed as a binary file. Otherwise, it is assumed to be a JSON file,
 * read, parsed, validated, and then passed to the WebAssembly module for
 * processing. It also handles dispatching status updates in case of errors.
 *
 * @param file The uploaded file to process.
 * @param traceviewerModule The initialized Trace Viewer v2 WASM module.
 * @param onFileProcessed Optional callback to execute when a file is
 *     successfully processed.
 */
async function processUploadedFile(
  file: File,
  traceviewerModule: TraceViewerV2Module,
  onFileProcessed?: () => void,
) {
  try {
    const arrayBuffer = await file.arrayBuffer();
    let uint8Array = new Uint8Array(arrayBuffer);

    if (hasGzipMagicNumber(uint8Array)) {
      const decompressed = await decompressGzip(uint8Array);
      if (!decompressed) {
        return;
      }
      uint8Array = new Uint8Array(decompressed.buffer as ArrayBuffer);
    }

    traceviewerModule.application.instance().restoreSavedTrackOrder?.();
    if (isPerfettoTrace(uint8Array, file.name)) {
      processPerfettoTrace(uint8Array, traceviewerModule);
    } else {
      processJsonTrace(uint8Array, traceviewerModule);
    }

    onFileProcessed?.();
  } catch (error) {
    dispatchErrorStatus(
      `Error processing file: ${
        error instanceof Error ? error.message : String(error)
      }`,
      error,
    );
  }
}

/**
 * Updates a URL object in-place with a `resolution` parameter based on the
 * canvas width.
 *
 * The resolution is calculated to optimize the number of trace events fetched
 * from the backend, preventing over-fetching of data that would not be visible.
 * If certain trace options are present (like filtering), resolution is set to 0
 * to fetch all data.
 *
 * @param urlObj The URL object to update with resolution parameter.
 * @param canvas The canvas element used to determine the viewer width.
 */
export function updateUrlWithResolution(
  urlObj: URL,
  canvas: HTMLCanvasElement | null | undefined,
): void {
  const params = urlObj.searchParams;

  // Default resolution to 0, which fetches all data.
  let resolution = 0;

  if (!params.has(TRACE_OPTIONS.SELECTED_GROUP_IDS)) {
    if (canvas) {
      const viewerWidth = canvas.clientWidth - HEADING_WIDTH;

      if (viewerWidth > 0) {
        // Calculate resolution based on the number of visual bins and multiply
        // by ZOOM_RATIO. This requests more data than strictly needed for the
        // current view (over-fetching), allowing the user to zoom in up to
        // ZOOM_RATIO times without losing detail (bins remain <=
        // MIN_EVENT_WIDTH in the zoomed view), avoiding immediate re-fetches.
        resolution = Math.round(viewerWidth / MIN_EVENT_WIDTH) * ZOOM_RATIO;
      }
    }
  }

  params.set(TRACE_VIEW_OPTION.RESOLUTION, resolution.toString());
}

function getTimeRangeFromUrl(urlObj: URL): [number, number] | undefined {
  const params = urlObj.searchParams;
  const viewStart = params.get(VIEW_START);
  const viewEnd = params.get(VIEW_END);
  if (viewStart && viewEnd) {
    const start = Number(viewStart);
    const end = Number(viewEnd);
    if (isFinite(start) && isFinite(end)) {
      return [start, end];
    }
  }
  return undefined;
}

/**
 * Expands the time range of the given URL using the pre-defined FETCH_RATIO,
 * with a minimum of MIN_FETCH_DURATION_MS. This must match the window that
 * `Timeline::InitializeLastFetchRequestRange` records as fetched.
 */
function expandUrlTimeRange(urlObj: URL, timeRange: [number, number]): void {
  const center = (timeRange[0] + timeRange[1]) / 2;
  const duration = timeRange[1] - timeRange[0];
  const fetchDuration = Math.max(duration * FETCH_RATIO, MIN_FETCH_DURATION_MS);
  const expandedStart = Math.max(0, center - fetchDuration / 2);
  const expandedEnd = center + fetchDuration / 2;

  urlObj.searchParams.set(
    TRACE_VIEW_OPTION.START_TIME_MS,
    String(expandedStart),
  );
  urlObj.searchParams.set(TRACE_VIEW_OPTION.END_TIME_MS, String(expandedEnd));
}

/**
 * Parses a raw `resolution` query parameter into a non-negative finite number,
 * or `undefined` when absent or malformed.
 *
 * Parsing philosophy and edge cases handled:
 * - Absent (`null`), empty (`""`), or whitespace-only (`"  "`) strings return
 *   `undefined` before `Number(...)` runs, because `Number("")` coerces to `0`
 *   and would otherwise silently trigger an unfiltered full-resolution load.
 * - Non-numeric (`"invalid"`), non-finite (`"Infinity"`), or negative (`"-1"`)
 *   values return `undefined` (backend `resolution` is an unsigned bin count).
 * - Valid numeric strings (such as `"0"`, whitespace-padded `" 0 "`, or
 *   scientific notation like `"1e3"`) are parsed to a `number` so callers can
 *   compare numerically and `sanitizeResolutionParam` can write `String(value)`
 *   back onto `urlObj.searchParams` as a canonical, trimmed numeric string.
 */
function parseFiniteResolution(value: string | null): number | undefined {
  const trimmed = value?.trim();
  if (!trimmed) {
    return undefined;
  }
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
}

/**
 * Extracts, validates, and normalizes the `resolution` `TraceViewOption` on
 * `urlObj` (falling back to the browser URL when absent or invalid on
 * `urlObj`), updating `urlObj.searchParams` in place to hold either a canonical
 * numeric string or no `resolution` parameter.
 *
 * Downstream data-loading behavior by return value:
 * - `0`: Explicit opt-in to full-resolution (`resolution=0`) initial loading.
 *   Unfiltered `.pb` loads stream the full session into the in-memory cache so
 *   subsequent zoom/pan navigation replays locally without network refetches.
 * - Positive finite `number` (`> 0`): Explicit event-bin downsampling count
 *   requested by the caller/URL; preserved on the initial request instead of
 *   overwriting with the canvas-derived resolution.
 * - `undefined`: No valid `resolution` override was supplied (`resolution` is
 *   deleted from `urlObj.searchParams`), so `fetchAndProcessTraceData` computes
 *   the default viewport-based resolution from the canvas width.
 */
function sanitizeResolutionParam(urlObj: URL): number | undefined {
  const browserUrl = new URL(window.location.href);
  const normalized =
    parseFiniteResolution(
      urlObj.searchParams.get(TRACE_VIEW_OPTION.RESOLUTION),
    ) ??
    parseFiniteResolution(
      browserUrl.searchParams.get(TRACE_VIEW_OPTION.RESOLUTION),
    );
  if (normalized === undefined) {
    urlObj.searchParams.delete(TRACE_VIEW_OPTION.RESOLUTION);
    return undefined;
  }
  urlObj.searchParams.set(TRACE_VIEW_OPTION.RESOLUTION, String(normalized));
  return normalized;
}

function propagateBrowserTraceOptions(urlObj: URL): void {
  const browserUrl = new URL(window.location.href);
  if (
    browserUrl.searchParams.get(TRACE_OPTIONS.MPMD_PIPELINE_VIEW_PARAM) ===
    'true'
  ) {
    urlObj.searchParams.set(TRACE_OPTIONS.MPMD_PIPELINE_VIEW_PARAM, 'true');
  }
}

// Fetches JSON data from the given URL. The `response.json()` method returns
// `any`, so this function returns `unknown`. Validation of the data structure
// (e.g., using `isTraceData`) is expected to be done by the caller.
async function loadJsonDataInternal(
  url: string,
  fetchData: TraceDataFetcher = fetch,
): Promise<unknown> {
  try {
    const response = await fetchData(url);
    if (!response.ok) {
      throw new Error(getFetchErrorMessage(response));
    }
    return await response.json();
  } catch (e) {
    if (!isAbortError(e)) {
      console.error('Failed to load JSON data:', e);
    }
    throw e;
  }
}

declare interface FetchDataEventDetail {
  start_time_ms: number;
  end_time_ms: number;
}

/**
 * The detail of an 'SearchEvents' custom event. The properties are quoted to
 * prevent renaming during minification.
 */
export declare interface SearchEventsEventDetail {
  events_query: string;
}

/**
 * Type guard for the 'SearchEvents' custom event.
 */
export function isSearchEventsEvent(
  event: Event,
): event is CustomEvent<SearchEventsEventDetail> {
  return (
    event instanceof CustomEvent &&
    event.detail &&
    typeof event.detail.events_query === 'string'
  );
}

function isFetchDataEvent(
  event: Event,
): event is CustomEvent<FetchDataEventDetail> {
  return (
    event instanceof CustomEvent &&
    event.detail &&
    typeof event.detail.start_time_ms === 'number' &&
    typeof event.detail.end_time_ms === 'number'
  );
}

const INCREMENTAL_EVENTS_PER_STEP = 131072;

declare interface IncrementalTraceViewerModule extends TraceViewerV2Module {
  beginIncrementalTraceEvents(timeRangeFromUrl?: [number, number]): void;
  appendCompressedTraceEventsIncremental(
    dataPtr: number,
    dataSize: number,
  ): void;
  stepIncrementalTraceEvents(maxEventsPerStep: number): boolean;
  finishIncrementalTraceEvents(): boolean | void;
}

function supportsIncrementalTraceEvents(
  module: TraceViewerV2Module,
): module is IncrementalTraceViewerModule {
  return (
    typeof module.beginIncrementalTraceEvents === 'function' &&
    typeof module.appendCompressedTraceEventsIncremental === 'function' &&
    typeof module.stepIncrementalTraceEvents === 'function' &&
    typeof module.finishIncrementalTraceEvents === 'function'
  );
}

function yieldToBrowserForNextSlice(): Promise<void> {
  return new Promise((resolve) => {
    if (
      typeof requestAnimationFrame === 'function' &&
      typeof document !== 'undefined' &&
      document.visibilityState !== 'hidden'
    ) {
      requestAnimationFrame(() => {
        setTimeout(resolve, 0);
      });
    } else {
      setTimeout(resolve, 0);
    }
  });
}

async function fetchAndProcessCompressedTraceDataIncremental({
  urlObj,
  traceviewerModule,
  isAbortRequested,
  abortSignal,
  timeRange,
  isRefetch = false,
  fetchData,
  createStreamWorker,
}: {
  urlObj: URL;
  traceviewerModule: IncrementalTraceViewerModule;
  isAbortRequested: () => boolean;
  abortSignal?: AbortSignal;
  timeRange?: [number, number];
  isRefetch?: boolean;
  fetchData?: TraceDataFetcher;
  createStreamWorker?: TraceStreamWorkerFactory;
}): Promise<boolean> {
  let sessionId = activeWasmIncrementalSessionId;
  let startedIncrementalSession = false;
  let settledIncrementalSession = false;
  const localAbortController = new AbortController();
  const forwardParentAbort = () => {
    localAbortController.abort();
  };
  if (abortSignal) {
    if (abortSignal.aborted) {
      localAbortController.abort();
    } else {
      abortSignal.addEventListener('abort', forwardParentAbort, {once: true});
    }
  }
  const isAborted = () =>
    isAbortRequested() || localAbortController.signal.aborted;

  const cancelIncrementalIfCurrent = () => {
    if (
      !settledIncrementalSession &&
      traceviewerModule === activeWasmModule &&
      sessionId === activeWasmIncrementalSessionId &&
      (startedIncrementalSession || isRefetch)
    ) {
      settledIncrementalSession = true;
      traceviewerModule.cancelIncrementalTraceEvents?.();
    }
  };

  let streamPromise: Promise<void> = Promise.resolve();

  try {
    const cacheKey = getCanonicalTraceCacheKey(urlObj);
    const hasInMemoryBlockCache =
      cachedFullTraceUrl === cacheKey &&
      typeof traceviewerModule.hasCachedTraceEventsIncremental === 'function' &&
      traceviewerModule.hasCachedTraceEventsIncremental();

    let effectiveTimeRange =
      timeRange &&
      timeRange.length === 2 &&
      isFinite(timeRange[0]) &&
      isFinite(timeRange[1]) &&
      timeRange[1] > timeRange[0]
        ? timeRange
        : undefined;
    if (
      isRefetch &&
      hasInMemoryBlockCache &&
      effectiveTimeRange &&
      effectiveTimeRange.length === 2
    ) {
      const requestedDurationMs = effectiveTimeRange[1] - effectiveTimeRange[0];
      const minWindowMs =
        requestedDurationMs > 0
          ? Math.min(1500, requestedDurationMs * (ZOOM_RATIO - 1))
          : 0;
      if (requestedDurationMs < minWindowMs) {
        const mid = 0.5 * (effectiveTimeRange[0] + effectiveTimeRange[1]);
        const halfWindow = 0.5 * minWindowMs;
        effectiveTimeRange = [Math.max(0, mid - halfWindow), mid + halfWindow];
      }
    }

    const yieldBetweenSlices = () => {
      if (isRefetch) {
        return new Promise<void>((resolve) => {
          setTimeout(resolve, 0);
        });
      }
      return yieldToBrowserForNextSlice();
    };

    if (isRefetch) {
      await yieldBetweenSlices();
      if (isAborted()) {
        cancelIncrementalIfCurrent();
        return false;
      }
    }

    if (
      isRefetch &&
      effectiveTimeRange === undefined &&
      typeof traceviewerModule.restoreFullTraceOverviewIfAvailable ===
        'function'
    ) {
      performance.mark('traceProcessStart');
      sessionId = ++activeWasmIncrementalSessionId;
      if (traceviewerModule.restoreFullTraceOverviewIfAvailable()) {
        settledIncrementalSession = true;
        return false;
      }
    }

    if (isRefetch && hasInMemoryBlockCache) {
      performance.mark('traceProcessStart');
      sessionId = ++activeWasmIncrementalSessionId;
      startedIncrementalSession = true;
      traceviewerModule.beginIncrementalTraceEvents(effectiveTimeRange);
      while (
        !isAborted() &&
        traceviewerModule.stepIncrementalTraceEvents(
          INCREMENTAL_EVENTS_PER_STEP * 4,
        )
      ) {
        await yieldBetweenSlices();
      }
      if (isAborted()) {
        cancelIncrementalIfCurrent();
        return false;
      }
      settledIncrementalSession = true;
      const fullLoaded = traceviewerModule.finishIncrementalTraceEvents();
      return typeof fullLoaded === 'boolean' ? fullLoaded : false;
    }

    const pendingFrames: Uint8Array[] = [];
    let streamFinished = false;
    let recordedStreamError: Error | null = null;
    const getStreamError = (): Error | null => recordedStreamError;
    let wakeConsumer: (() => void) | null = null;
    const isFullSessionStream =
      !isRefetch &&
      urlObj.searchParams.get(TRACE_VIEW_OPTION.RESOLUTION) === '0' &&
      !urlObj.searchParams.has(TRACE_VIEW_OPTION.START_TIME_MS) &&
      !urlObj.searchParams.has(TRACE_VIEW_OPTION.END_TIME_MS);
    const bypassOpfsCache =
      !isFullSessionStream || shouldBypassOpfsCache(urlObj);
    const persistOpfsCache =
      isFullSessionStream && !isClientOpfsBypassRequested(urlObj);

    // Use cached full-stream compressed frames on zoom/pan refetches when
    // available so zooming in or out on huge profiles takes ~150ms in memory
    // with zero backend round-trip.
    const canReplayFromCache =
      isRefetch &&
      cachedFullTraceUrl === cacheKey &&
      cachedFullTraceFrames !== null &&
      cachedFullTraceFrames.length > 0;

    const collectedFrames: Uint8Array[] = [];
    let collectedBytes = 0;
    let canCacheFullFrames = isFullSessionStream;

    const notifyConsumer = () => {
      if (wakeConsumer) {
        const fn = wakeConsumer;
        wakeConsumer = null;
        fn();
      }
    };
    localAbortController.signal.addEventListener('abort', notifyConsumer, {
      once: true,
    });

    if (canReplayFromCache && cachedFullTraceFrames) {
      pendingFrames.push(...cachedFullTraceFrames);
      streamFinished = true;
    } else {
      streamPromise = streamCompressedTraceFrames({
        urlObj,
        cacheKey,
        bypassCache: bypassOpfsCache,
        persistCache: persistOpfsCache,
        onFrame: (frame: Uint8Array) => {
          if (frame.byteLength === 0) return;
          pendingFrames.push(frame);
          if (canCacheFullFrames) {
            if (
              collectedBytes + frame.byteLength <=
              MAX_CACHED_COMPRESSED_BYTES
            ) {
              collectedFrames.push(frame);
              collectedBytes += frame.byteLength;
            } else {
              canCacheFullFrames = false;
              collectedFrames.length = 0;
            }
          }
          notifyConsumer();
        },
        isAbortRequested: isAborted,
        abortSignal: localAbortController.signal,
        fetchData,
        createStreamWorker,
      })
        .then(() => {
          if (
            !isAborted() &&
            canCacheFullFrames &&
            collectedFrames.length > 0
          ) {
            cachedFullTraceUrl = cacheKey;
            cachedFullTraceFrames = collectedFrames;
          }
          streamFinished = true;
          notifyConsumer();
        })
        .catch((err: unknown) => {
          recordedStreamError =
            err instanceof Error ? err : new Error(String(err));
          streamFinished = true;
          notifyConsumer();
        });

      while (
        pendingFrames.length === 0 &&
        !streamFinished &&
        !isAborted()
      ) {
        await new Promise<void>((resolve) => {
          wakeConsumer = resolve;
        });
      }
    }

    const initialStreamErr = getStreamError();
    if (initialStreamErr !== null) {
      throw initialStreamErr;
    }
    if (isAborted()) {
      cancelIncrementalIfCurrent();
      return false;
    }

    if (!canReplayFromCache) {
      window.dispatchEvent(
        new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
          detail: {status: TraceViewerV2LoadingStatus.PROCESSING_DATA},
        }),
      );

      // Yield to the event loop so the UI can display 'Processing data' before
      // starting the incremental processing pipeline.
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    }

    if (isAborted()) {
      cancelIncrementalIfCurrent();
      return false;
    }

    performance.mark('traceProcessStart');
    sessionId = ++activeWasmIncrementalSessionId;
    startedIncrementalSession = true;
    traceviewerModule.beginIncrementalTraceEvents(effectiveTimeRange);

    let hasMoreInWasm = false;
    let firstSliceYielded = false;
    let enqueuedEventFrames = 0;
    let initialOverlayDismissed = isRefetch;

    const enqueueNextPendingFrameIntoWasm = () => {
      const maxFramesToEnqueue = canReplayFromCache
        ? 2
        : firstSliceYielded
          ? 4
          : 1;
      let enqueued = 0;
      while (pendingFrames.length > 0 && enqueued < maxFramesToEnqueue) {
        const frame = pendingFrames.shift();
        if (!frame) break;
        if (frame.byteLength === 0) continue;
        if (
          canReplayFromCache &&
          typeof traceviewerModule.shouldEnqueueCompressedFrameIncremental ===
            'function' &&
          !traceviewerModule.shouldEnqueueCompressedFrameIncremental()
        ) {
          continue;
        }
        let dataPtr: number | undefined;
        try {
          dataPtr = traceviewerModule._malloc(frame.byteLength);
          if (!dataPtr) {
            throw new Error('Failed to allocate WASM memory buffer');
          }
          traceviewerModule.HEAPU8.set(frame, dataPtr);
          traceviewerModule.appendCompressedTraceEventsIncremental(
            dataPtr,
            frame.byteLength,
          );
          // Header-only frames (timespan metadata only, < 256 bytes) do not
          // contain track events and should not consume the initial frame slot
          // unless it is the only remaining frame.
          if (frame.byteLength >= 256 || pendingFrames.length === 0) {
            enqueued++;
            enqueuedEventFrames++;
          }
        } finally {
          if (dataPtr !== undefined && dataPtr !== 0) {
            traceviewerModule._free(dataPtr);
          }
        }
      }
    };

    while (!isAborted()) {
      const loopStreamErr = getStreamError();
      if (loopStreamErr !== null) {
        throw loopStreamErr;
      }
      if (!hasMoreInWasm) {
        enqueueNextPendingFrameIntoWasm();
      }
      const stepBudget = isRefetch
        ? INCREMENTAL_EVENTS_PER_STEP
        : INCREMENTAL_EVENTS_PER_STEP * 2;
      hasMoreInWasm = traceviewerModule.stepIncrementalTraceEvents(stepBudget);
      if (enqueuedEventFrames > 0) {
        firstSliceYielded = true;
        if (
          !initialOverlayDismissed &&
          (hasMoreInWasm || pendingFrames.length > 0)
        ) {
          initialOverlayDismissed = true;
          // Transition out of initial full-screen loading overlay once the first
          // event slice is flushed to the canvas; keep toolbar spinner active if
          // additional frames are still streaming.
          window.dispatchEvent(
            new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
              detail: {status: TraceViewerV2LoadingStatus.IDLE},
            }),
          );
        }
      }
      if (hasMoreInWasm || pendingFrames.length > 0) {
        await yieldBetweenSlices();
        continue;
      }
      if (streamFinished || isAborted()) {
        break;
      }
      await new Promise<void>((resolve) => {
        wakeConsumer = resolve;
      });
    }

    await streamPromise;
    const postStreamErr = getStreamError();
    if (postStreamErr !== null) {
      throw postStreamErr;
    }
    while (!isAborted() && (hasMoreInWasm || pendingFrames.length > 0)) {
      if (!hasMoreInWasm) {
        enqueueNextPendingFrameIntoWasm();
      }
      hasMoreInWasm = traceviewerModule.stepIncrementalTraceEvents(
        isRefetch
          ? INCREMENTAL_EVENTS_PER_STEP
          : INCREMENTAL_EVENTS_PER_STEP * 4,
      );
      if (hasMoreInWasm || pendingFrames.length > 0) {
        await yieldBetweenSlices();
      }
    }
    if (isAborted()) {
      cancelIncrementalIfCurrent();
      return false;
    }
    settledIncrementalSession = true;
    const fullLoaded = traceviewerModule.finishIncrementalTraceEvents();
    if (
      !isRefetch &&
      typeof traceviewerModule.hasCachedTraceEventsIncremental === 'function' &&
      traceviewerModule.hasCachedTraceEventsIncremental()
    ) {
      cachedFullTraceUrl = cacheKey;
      cachedFullTraceFrames = null;
    }
    return typeof fullLoaded === 'boolean' ? fullLoaded : !isRefetch;
  } catch (e) {
    localAbortController.abort();
    cancelIncrementalIfCurrent();
    throw e;
  } finally {
    if (abortSignal) {
      abortSignal.removeEventListener('abort', forwardParentAbort);
    }
    localAbortController.abort();
    await Promise.allSettled([streamPromise]);
  }
}

/**
 * Fetches and processes trace data (either .pb or .json) from the given URL.
 * Handles status dispatches, yielding to event loop, and latency measurement.
 * Returns `false` if the load was aborted, superseded, or failed, and `true`
 * otherwise.
 */
async function fetchAndProcessTraceData({
  urlObj,
  traceviewerModule,
  isAbortRequested,
  abortSignal,
  timeRange,
  isRefetch = false,
  errorContext,
  fetchData,
  createStreamWorker,
  onFullTraceLoaded,
}: {
  urlObj: URL;
  traceviewerModule: TraceViewerV2Module;
  isAbortRequested: () => boolean;
  abortSignal?: AbortSignal;
  timeRange?: [number, number];
  isRefetch?: boolean;
  errorContext: string;
  fetchData?: TraceDataFetcher;
  createStreamWorker?: TraceStreamWorkerFactory;
  onFullTraceLoaded?: () => void;
}): Promise<boolean> {
  if (!isRefetch) {
    window.dispatchEvent(
      new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {status: TraceViewerV2LoadingStatus.LOADING_DATA},
      }),
    );
  }

  if (isRefetch || !urlObj.searchParams.has(TRACE_VIEW_OPTION.RESOLUTION)) {
    updateUrlWithResolution(urlObj, traceviewerModule.canvas);
  }

  try {
    if (
      urlObj.pathname.endsWith('.pb') ||
      urlObj.searchParams.get('format') === 'pb'
    ) {
      const isUnfilteredFullResolutionLoad =
        urlObj.searchParams.get(TRACE_VIEW_OPTION.RESOLUTION) === '0' &&
        !urlObj.searchParams.has(TRACE_VIEW_OPTION.START_TIME_MS) &&
        !urlObj.searchParams.has(TRACE_VIEW_OPTION.END_TIME_MS);
      if (
        supportsIncrementalTraceEvents(traceviewerModule) &&
        (isRefetch || isUnfilteredFullResolutionLoad)
      ) {
        const fullLoaded = await fetchAndProcessCompressedTraceDataIncremental({
          urlObj,
          traceviewerModule,
          isAbortRequested,
          abortSignal,
          timeRange,
          isRefetch,
          fetchData,
          createStreamWorker,
        });
        if (isAbortRequested()) return false;
        if (fullLoaded) {
          onFullTraceLoaded?.();
        }
      } else {
        const buffer = await loadCompressedTraceDataInternal(
          urlObj.toString(),
          fetchData,
        );
        if (isAbortRequested()) return false;

        window.dispatchEvent(
          new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
            detail: {status: TraceViewerV2LoadingStatus.PROCESSING_DATA},
          }),
        );

        // Yield to the event loop to allow the UI to re-render and display
        // the 'Processing data' status before the potentially long-running
        // processCompressedTraceEvents call.
        await new Promise((resolve) => {
          setTimeout(resolve, 0);
        });

        if (isAbortRequested()) return false;

        performance.mark('traceProcessStart');

        let dataPtr: number | undefined;
        try {
          dataPtr = traceviewerModule._malloc(buffer.byteLength);
          if (!dataPtr) {
            throw new Error('Failed to allocate WASM memory buffer');
          }
          traceviewerModule.HEAPU8.set(new Uint8Array(buffer), dataPtr);
          traceviewerModule.processCompressedTraceEvents(
            dataPtr,
            buffer.byteLength,
            timeRange,
          );
        } finally {
          if (dataPtr !== undefined && dataPtr !== 0) {
            traceviewerModule._free(dataPtr);
          }
        }
      }
    } else {
      const jsonData = await loadJsonDataInternal(urlObj.toString(), fetchData);
      if (isAbortRequested()) return false;

      if (!isTraceData(jsonData)) {
        console.error('File does not contain valid trace events.');
        window.dispatchEvent(
          new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
            detail: {status: TraceViewerV2LoadingStatus.IDLE},
          }),
        );
        return true;
      }

      maybeDispatchDetailsReceivedEvent(jsonData);

      window.dispatchEvent(
        new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
          detail: {status: TraceViewerV2LoadingStatus.PROCESSING_DATA},
        }),
      );

      // Yield to the event loop to allow the UI to re-render and display
      // the 'Processing data' status before the potentially long-running
      // processTraceEvents call.
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });

      if (isAbortRequested()) return false;

      performance.mark('traceProcessStart');

      traceviewerModule.processTraceEvents(jsonData, timeRange);
    }

    performance.mark('traceProcessEnd');
    performance.measure(
      'traceProcessingTime',
      'traceProcessStart',
      'traceProcessEnd',
    );

    // HEAPU8.length represents the total size of the WASM heap (the memory
    // reserved from the browser). Since ALLOW_MEMORY_GROWTH is enabled,
    // this value will effectively track the peak memory reservation reached
    // during processing. This is a good metric, but worth noting it might
    // differ from the actual active allocation size.
    window.wasmMemoryBytes = traceviewerModule.HEAPU8
      ? traceviewerModule.HEAPU8.length
      : 0;

    window.dispatchEvent(
      new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {status: TraceViewerV2LoadingStatus.IDLE},
      }),
    );
    return true;
  } catch (e) {
    if (isAbortRequested()) {
      return false;
    }
    if (isAbortError(e)) {
      window.dispatchEvent(
        new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
          detail: {status: TraceViewerV2LoadingStatus.IDLE},
        }),
      );
      return false;
    }
    console.error(`Error ${errorContext}:`, e);
    const errorMessage = e instanceof Error ? e.message : String(e);
    window.dispatchEvent(
      new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {
          status: TraceViewerV2LoadingStatus.ERROR,
          message: errorMessage,
        },
      }),
    );
    return true;
  }
}

async function handleFetchDataEvent(
  event: Event,
  getCurrentDataUrl: () => string | null,
  traceviewerModule: TraceViewerV2Module | null,
  isFullTraceLoaded: () => boolean = () => false,
  isCurrentlyLoading: () => boolean = () => false,
  isRefetchSuperseded: () => boolean = () => false,
  abortSignal?: AbortSignal,
  createStreamWorker?: TraceStreamWorkerFactory,
) {
  if (!isFetchDataEvent(event)) {
    return;
  }
  if (isFullTraceLoaded() || isCurrentlyLoading()) {
    return;
  }
  const detail = event.detail;
  const initialDataUrl = getCurrentDataUrl();
  if (!initialDataUrl) {
    console.warn('Data URL not set, cannot fetch new data.');
    return;
  }
  if (!traceviewerModule) {
    console.warn('Trace viewer module not initialized.');
    return;
  }

  const urlObj = new URL(initialDataUrl, window.location.href);
  urlObj.searchParams.delete('use_saved_result');
  propagateBrowserTraceOptions(urlObj);

  urlObj.searchParams.set(
    TRACE_VIEW_OPTION.START_TIME_MS,
    String(detail.start_time_ms),
  );
  urlObj.searchParams.set(
    TRACE_VIEW_OPTION.END_TIME_MS,
    String(detail.end_time_ms),
  );

  await fetchAndProcessTraceData({
    urlObj,
    traceviewerModule,
    isAbortRequested: () =>
      initialDataUrl !== getCurrentDataUrl() ||
      traceviewerModule !== activeWasmModule ||
      isRefetchSuperseded(),
    abortSignal,
    timeRange: [detail.start_time_ms, detail.end_time_ms],
    isRefetch: true,
    errorContext: 'fetching new data',
    createStreamWorker,
  });
}

/**
 * Options for Trace Viewer v2 initialization.
 */
export declare interface TraceViewerV2Options {
  // Optional callback to execute when a file is successfully uploaded to the
  // application.
  onFileUploadedToXprof?: () => void;
  // Optional factory used to construct the dedicated trace-streaming Web
  // Worker during URL loads and zoom/pan refetches.
  createStreamWorker?: TraceStreamWorkerFactory;
}

/**
 * Initializes the Trace Viewer v2 application.
 * This function sets up the necessary environment, including requesting a
 * WebGPU device, configuring a canvas for WebGPU rendering, and loading the
 * WebAssembly module for the trace viewer. It also exposes a method on the
 * returned module to load trace data from a JSON URL. This is the main entry
 * point for the Trace Viewer v2.
 *
 * @param options Options for configuring the Trace Viewer v2 module.
 * @return A promise that resolves with the initialized TraceViewerV2Module, or
 *     null if initialization fails.
 */
export async function traceViewerV2Main(
  options?: TraceViewerV2Options,
): Promise<TraceViewerV2Module | null> {
  // Shut down any existing WASM application and clean up event listeners
  // before starting a new one. This prevents leaking resources and having
  // multiple active instances fighting for the canvas or processing duplicate
  // events.
  shutdownTraceViewerV2();

  // Define getFeatureFlag on window so C++ can access it during initialization
  // before the WASM module fully loads.
  window.getFeatureFlag = (name: string): boolean => {
    // TODO(b/498744795): Now only supports boolean flags (true/false).
    // Will be extended in the future.
    try {
      const value = window.localStorage.getItem(`xprof_ff_${name}`);
      if (value !== null) {
        return value === 'true';
      }
    } catch (e) {
      // Handle potential SecurityError when accessing localStorage.
      console.warn(`Failed to read feature flag ${name} from localStorage:`, e);
    }
    // If the flag is not in local storage, use the default value.
    return getDefaultFeatureFlag(name);
  };

  let traceviewerModule: TraceViewerV2Module | null = null;
  let currentDataUrl: string | null = null;
  let currentLoadingPromise: Promise<void> | null = null;
  let currentStreamWorkerFactory: TraceStreamWorkerFactory | undefined =
    options?.createStreamWorker;
  let fullTraceLoaded = false;
  let activeRefetchId = 0;
  let hasLoadedDataset = false;

  try {
    traceviewerModule = await initGpuAndStartWasmApp();
    traceviewerModule.getFeatureFlag = window.getFeatureFlag;
    activeWasmModule = traceviewerModule;
  } catch (e) {
    const errorMessage = e instanceof Error ? e.message : String(e);
    console.error('Application Initialization Failed:', e);
    window.dispatchEvent(
      new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {
          status: TraceViewerV2LoadingStatus.ERROR,
          message: errorMessage,
        },
      }),
    );
    return null;
  }

  setupFileInputHandler(traceviewerModule, () => {
    activeLoadAbortController?.abort();
    activeLoadAbortController = null;
    activeRefetchAbortController?.abort();
    activeRefetchAbortController = null;
    currentDataUrl = null;
    currentStreamWorkerFactory = options?.createStreamWorker;
    fullTraceLoaded = false;
    clearCachedFullTrace();
    ++activeRefetchId;
    hasLoadedDataset = true;
    if (options?.onFileUploadedToXprof) {
      options.onFileUploadedToXprof();
    }
  });

  const resizeObserver = new ResizeObserver(() => {
    if (traceviewerModule?.canvas) {
      requestAnimationFrame(() => {
        // We use clientWidth/clientHeight to get the logical (CSS) pixel size
        // of the canvas element.
        const width = traceviewerModule.canvas.clientWidth;
        const height = traceviewerModule.canvas.clientHeight;
        if (width === 0 || height === 0) {
          return;
        }
        const dpr = window.devicePixelRatio;
        traceviewerModule.application.instance().resize(dpr, width, height);
      });
    }
  });
  resizeObserver.observe(traceviewerModule.canvas);
  // Track the resize observer to disconnect it on shutdown if needed. For now
  // it's tied to the canvas element.

  // Add a method to the module to load data from a URL
  traceviewerModule.loadTraceData = async (
    url: string,
    loadOptions?: LoadTraceDataOptions,
  ) => {
    if (url === currentDataUrl && currentLoadingPromise) {
      return currentLoadingPromise;
    }
    activeLoadAbortController?.abort();
    activeRefetchAbortController?.abort();
    activeRefetchAbortController = null;
    const loadAbortController = new AbortController();
    activeLoadAbortController = loadAbortController;
    currentDataUrl = url;
    const streamWorkerFactory =
      loadOptions?.createStreamWorker ?? options?.createStreamWorker;
    currentStreamWorkerFactory = streamWorkerFactory;
    fullTraceLoaded = false;
    clearCachedFullTrace();
    ++activeRefetchId;
    if (!traceviewerModule) return;
    if (hasLoadedDataset) {
      if (typeof traceviewerModule.ResetDataset === 'function') {
        ++activeWasmIncrementalSessionId;
        traceviewerModule.ResetDataset();
      }
    }
    hasLoadedDataset = true;

    let settled = false;
    let completed = false;
    const loadingPromise = (async () => {
      try {
        let urlObj: URL;
        let explicitResolution: number | undefined;
        try {
          urlObj = new URL(url, window.location.href);
          propagateBrowserTraceOptions(urlObj);
          explicitResolution = sanitizeResolutionParam(urlObj);
        } catch (e) {
          console.error('Invalid URL:', url, e);
          const errorMessage = (e as Error).message;
          window.dispatchEvent(
            new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
              detail: {
                status: TraceViewerV2LoadingStatus.ERROR,
                message: errorMessage,
              },
            }),
          );
          return;
        }

        const timeRangeFromUrl = getTimeRangeFromUrl(urlObj);
        const usesFullSessionIncrementalPb =
          explicitResolution === 0 &&
          (urlObj.pathname.endsWith('.pb') ||
            urlObj.searchParams.get('format') === 'pb') &&
          supportsIncrementalTraceEvents(traceviewerModule);
        if (timeRangeFromUrl && !usesFullSessionIncrementalPb) {
          expandUrlTimeRange(urlObj, timeRangeFromUrl);
        }

        traceviewerModule.application.instance().restoreSavedTrackOrder?.();

        completed = await fetchAndProcessTraceData({
          urlObj,
          traceviewerModule,
          isAbortRequested: () =>
            loadAbortController.signal.aborted ||
            url !== currentDataUrl ||
            traceviewerModule !== activeWasmModule,
          abortSignal: loadAbortController.signal,
          timeRange: timeRangeFromUrl,
          errorContext: 'loading data',
          fetchData: loadOptions?.fetchData,
          createStreamWorker: streamWorkerFactory,
          onFullTraceLoaded: () => {
            fullTraceLoaded = true;
          },
        });
      } finally {
        settled = true;
        if (activeLoadAbortController === loadAbortController) {
          activeLoadAbortController = null;
        }
        if (url === currentDataUrl) {
          currentLoadingPromise = null;
          if (!completed) {
            currentDataUrl = null;
          }
        }
      }
    })();
    if (!settled) {
      currentLoadingPromise = loadingPromise;
    }

    return loadingPromise;
  };

  traceviewerModule.loadSearchResults = async (url: string) => {
    if (!traceviewerModule) return;
    try {
      let urlObj: URL;
      try {
        urlObj = new URL(url, window.location.href);
        propagateBrowserTraceOptions(urlObj);
      } catch (e) {
        console.error('Invalid URL for search results:', url, e);
        return;
      }

      if (
        urlObj.pathname.endsWith('.pb') ||
        urlObj.searchParams.get('format') === 'pb'
      ) {
        const buffer = await loadCompressedTraceDataInternal(urlObj.toString());
        traceviewerModule.setCompressedSearchResultsInWasm(
          new Uint8Array(buffer),
        );
      } else {
        const jsonData = (await loadJsonDataInternal(
          urlObj.toString(),
        )) as Record<string, unknown>;
        // Mirror the legacy behavior: gently default to an empty array if
        // traceEvents is missing so that the WASM module can safely clear out
        // any previous search results.
        const normalizedData: TraceData = {
          ...jsonData,
          traceEvents: Array.isArray(jsonData?.['traceEvents'])
            ? (jsonData['traceEvents'] as Array<{[key: string]: unknown}>)
            : [],
        };
        traceviewerModule.setSearchResultsInWasm(normalizedData);
      }
    } catch (e) {
      console.error('Error loading search results:', e);
      throw e;
    }
  };

  registerWindowListener(FETCH_DATA_EVENT_NAME, (event: Event) => {
    if (
      !isFetchDataEvent(event) ||
      fullTraceLoaded ||
      currentLoadingPromise !== null
    ) {
      return;
    }
    activeRefetchAbortController?.abort();
    const refetchAbortController = new AbortController();
    activeRefetchAbortController = refetchAbortController;
    const refetchId = ++activeRefetchId;
    void handleFetchDataEvent(
      event,
      () => currentDataUrl,
      traceviewerModule,
      () => fullTraceLoaded,
      () => currentLoadingPromise !== null,
      () =>
        refetchAbortController.signal.aborted || refetchId !== activeRefetchId,
      refetchAbortController.signal,
      currentStreamWorkerFactory,
    ).finally(() => {
      if (activeRefetchAbortController === refetchAbortController) {
        activeRefetchAbortController = null;
      }
    });
  });

  return traceviewerModule;
}
