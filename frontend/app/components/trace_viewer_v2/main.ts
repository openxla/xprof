
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
  }
}

  export declare interface TraceViewerV2Module extends EmscriptenModule {
  HEAPU8: Uint8Array;
  canvas: HTMLCanvasElement;
  callMain(args: string[]): void;
  preinitializedWebGPUDevice: GPUDevice | null;
  processTraceEvents(
    data: TraceData,
    timeRangeFromUrl?: [number, number],
  ): void;
  getAllFlowCategories(): Array<{id: number; name: string}>;
  setSearchResultsInWasm(data: TraceData): void;
  loadJsonData?(url: string): Promise<void>;
  StringVector: {
    size(): number;
    get(index: number): string;
    toArray(): string[];
  };
  IntVector: {size(): number; get(index: number): number};
  application: {
    instance(): {
      shutdown(): void;
      dataProvider(): {getFlowCategories(): TraceViewerV2Module['IntVector']};
      getCurrentSearchResultIndex(): number;
      getSearchResultsCount(): number;
      navigateToNextSearchResult(): void;
      navigateToPrevSearchResult(): void;
      resize(dpr: number, width: number, height: number): void;
      setSearchQuery(query: string): void;
      setMouseMode(mode: number): void;
      setVisibleFlowCategory(categoryId: number): void;
      setVisibleFlowCategories(categoryIds: number[]): void;
    };
  };
}

/**
 * Interface for the trace data loaded by the trace viewer.
 */
export declare interface TraceData {
  traceEvents: Array<{[key: string]: unknown}>;
  fullTimespan?: [number, number];
  details?: TraceDetails;
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
    data.hasOwnProperty('traceEvents') &&
    Array.isArray((data as TraceData).traceEvents)
  );
}

/**
 * Dispatches a DETAILS_RECEIVED_EVENT if the trace data contains details.
 */
function maybeDispatchDetailsReceivedEvent(jsonData: TraceData) {
  const rawDetails = jsonData.details as unknown;
  if (!rawDetails) return;

  const details: TraceDetails = new Map();
  if (Array.isArray(rawDetails)) {
    for (const d of rawDetails) {
      if (d && typeof d === 'object' && d.name === 'full_dma') {
        details.set('full_dma', d.value);
      }
    }
  } else if (typeof rawDetails === 'object' && rawDetails !== null) {
    const rawDetailsObj = rawDetails as Record<string, unknown>;
    if (rawDetailsObj['full_dma'] !== undefined) {
      details.set('full_dma', rawDetailsObj['full_dma'] as boolean);
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

/**
 * Interface for tracking event listeners registered on the window.
 * Each instance stores the event `type` and the `listener` function,
 * allowing them to be properly removed when the trace viewer is shut down.
 */
interface RegisteredListener {
  type: string;
  listener: EventListener;
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

async function getWebGpuDevice(): Promise<GPUDevice> {
  const gpu = navigator.gpu;
  if (!gpu) {
    throw new Error('WebGPU not supported on this browser.');
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    throw new Error('WebGPU cannot be initialized- adapter not found');
  }
  const device = await adapter.requestDevice();
  if (!device) {
    throw new Error(
      'WebGPU cannot be initialized - failed to get WebGPU device.',
    );
  }
  // tslint:disable-next-line:no-any
  (device as any).lost
    .then(() => {
      throw new Error('WebGPU Cannot be initialized - Device has been lost');
    })
    .catch(() => {});
  return device;
}

function configureCanvas(canvas: HTMLCanvasElement, device: GPUDevice) {
  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new Error('Context not found for canvas.');
  }
  context.configure({
    device,
    format: navigator.gpu.getPreferredCanvasFormat(),
  });
}

async function loadAndStartWasm(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
): Promise<TraceViewerV2Module> {
  const moduleConfig = {
    canvas,
    print: console.log,
    printErr: console.error,
    setStatus: console.debug,
    noInitialRun: true,
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
    const existingScript = document.querySelector(
      'script[src*="trace_viewer_v2.js"]',
    );
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
function dispatchErrorStatus(msg: string) {
  console.error(msg);

  window.dispatchEvent(
    new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
      detail: {
        status: TraceViewerV2LoadingStatus.ERROR,
        message: msg,
      },
    }),
  );
}

/**
 * Processes an uploaded file containing trace data.
 *
 * Reads the file content, parses it as JSON, validates the data structure, and
 * passes the valid trace events to the provided WebAssembly module for
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
    const fileContent = await file.text();
    const jsonData = JSON.parse(fileContent) as unknown;

    if (!isTraceData(jsonData)) {
      dispatchErrorStatus('File does not contain valid trace events.');
      return;
    }

    traceviewerModule.processTraceEvents(jsonData, undefined);

    onFileProcessed?.();
  } catch (error) {
    dispatchErrorStatus(
      `Error processing file: ${error instanceof Error ? error.message : String(error)}`,
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
 * Expands the time range of the given URL using the pre-defined FETCH_RATIO.
 */
function expandUrlTimeRange(urlObj: URL, timeRange: [number, number]): void {
  const center = (timeRange[0] + timeRange[1]) / 2;
  const duration = timeRange[1] - timeRange[0];
  const expandedStart = center - (duration * FETCH_RATIO) / 2;
  const expandedEnd = center + (duration * FETCH_RATIO) / 2;

  urlObj.searchParams.set(
    TRACE_VIEW_OPTION.START_TIME_MS,
    String(expandedStart),
  );
  urlObj.searchParams.set(TRACE_VIEW_OPTION.END_TIME_MS, String(expandedEnd));
}

// Fetches JSON data from the given URL. The `response.json()` method returns
// `any`, so this function returns `unknown`. Validation of the data structure
// (e.g., using `isTraceData`) is expected to be done by the caller.
async function loadJsonDataInternal(url: string): Promise<unknown> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (e) {
    console.error('Failed to load JSON data:', e);
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

async function handleFetchDataEvent(
  event: Event,
  getCurrentDataUrl: () => string | null,
  traceviewerModule: TraceViewerV2Module | null,
) {
  if (!isFetchDataEvent(event)) {
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

  try {
    window.dispatchEvent(
      new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {status: TraceViewerV2LoadingStatus.LOADING_DATA},
      }),
    );

    const urlObj = new URL(initialDataUrl, window.location.href);

    urlObj.searchParams.set(
      TRACE_VIEW_OPTION.START_TIME_MS,
      String(detail.start_time_ms),
    );
    urlObj.searchParams.set(
      TRACE_VIEW_OPTION.END_TIME_MS,
      String(detail.end_time_ms),
    );

    // Update resolution
    updateUrlWithResolution(urlObj, traceviewerModule.canvas);

    // TODO(b/470214911): Add support for additional query parameters to allow
    // for filtering by specific events, groups, or other criteria.
    const jsonData = await loadJsonDataInternal(urlObj.toString());

    if (initialDataUrl !== getCurrentDataUrl()) {
      return;
    }

    if (!isTraceData(jsonData)) {
      console.error('File does not contain valid trace events.');
      window.dispatchEvent(
        new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
          detail: {status: TraceViewerV2LoadingStatus.IDLE},
        }),
      );
      return;
    }

    maybeDispatchDetailsReceivedEvent(jsonData);

    window.dispatchEvent(
      new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {status: TraceViewerV2LoadingStatus.PROCESSING_DATA},
      }),
    );

    // Yield to the event loop to allow the UI to re-render and display the
    // 'Processing data' status before the potentially long-running
    // processTraceEvents call.
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    if (initialDataUrl !== getCurrentDataUrl()) {
      return;
    }

    performance.mark('traceProcessStart');

    traceviewerModule.processTraceEvents(
      jsonData,
      /* timeRangeFromUrl= */ undefined,
    );

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
  } catch (e) {
    console.error('Error fetching new data:', e);
    const error = e as Error;
    window.dispatchEvent(
      new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {
          status: TraceViewerV2LoadingStatus.ERROR,
          message: error.message,
        },
      }),
    );
  }
}

/**
 * Options for Trace Viewer v2 initialization.
 */
export declare interface TraceViewerV2Options {
  // Optional callback to execute when a file is successfully uploaded to the application.
  onFileUploadedToXprof?: () => void;
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

  let traceviewerModule: TraceViewerV2Module | null = null;
  let currentDataUrl: string | null = null;
  let currentLoadingPromise: Promise<void> | null = null;

  try {
    traceviewerModule = await initGpuAndStartWasmApp();
    activeWasmModule = traceviewerModule;
  } catch (e) {
    const error = e as Error;
    console.error('Application Initialization Failed:', error);
    window.dispatchEvent(
      new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {
          status: TraceViewerV2LoadingStatus.ERROR,
          message: error.message,
        },
      }),
    );
    return null;
  }

  setupFileInputHandler(traceviewerModule, () => {
    currentDataUrl = null;
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
  traceviewerModule.loadJsonData = async (url: string) => {
    if (url === currentDataUrl && currentLoadingPromise) {
      return currentLoadingPromise;
    }
    currentDataUrl = url;
    if (!traceviewerModule) return;

    currentLoadingPromise = (async () => {
      try {
        window.dispatchEvent(
          new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
            detail: {status: TraceViewerV2LoadingStatus.LOADING_DATA},
          }),
        );

        let urlObj: URL;
        try {
          urlObj = new URL(url, window.location.href);
        } catch (e) {
          console.error('Invalid URL:', url, e);

          const error = e as Error;

          window.dispatchEvent(
            new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
              detail: {
                status: TraceViewerV2LoadingStatus.ERROR,
                message: error.message,
              },
            }),
          );
          return;
        }

        const timeRangeFromUrl = getTimeRangeFromUrl(urlObj);
        if (timeRangeFromUrl) {
          expandUrlTimeRange(urlObj, timeRangeFromUrl);
        }

        updateUrlWithResolution(urlObj, traceviewerModule.canvas);

        const jsonData = await loadJsonDataInternal(urlObj.toString());

        if (url !== currentDataUrl) {
          return;
        }

        if (!isTraceData(jsonData)) {
          console.error('File does not contain valid trace events.');

          window.dispatchEvent(
            new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
              detail: {status: TraceViewerV2LoadingStatus.IDLE},
            }),
          );

          return;
        }

        maybeDispatchDetailsReceivedEvent(jsonData);

        window.dispatchEvent(
          new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
            detail: {status: TraceViewerV2LoadingStatus.PROCESSING_DATA},
          }),
        );

        // Yield to the event loop to allow the UI to re-render and display the
        // 'Processing data' status before the potentially long-running
        // processTraceEvents call.
        await new Promise((resolve) => {
          setTimeout(resolve, 0);
        });

        performance.mark('traceProcessStart');

        traceviewerModule.processTraceEvents(jsonData, timeRangeFromUrl);

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
      } catch (e) {
        console.error('Error processing file:', e);
        const error = e as Error;
        window.dispatchEvent(
          new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
            detail: {
              status: TraceViewerV2LoadingStatus.ERROR,
              message: error.message,
            },
          }),
        );
      } finally {
        if (url === currentDataUrl) {
          currentLoadingPromise = null;
        }
      }
    })();

    return currentLoadingPromise;
  };

  registerWindowListener(FETCH_DATA_EVENT_NAME, (event: Event) => {
    handleFetchDataEvent(event, () => currentDataUrl, traceviewerModule);
  });

  return traceviewerModule;
}

// Expose to window for integration tests.
// tslint:disable-next-line:no-any
(window as any)['traceViewerV2Main'] = traceViewerV2Main;
