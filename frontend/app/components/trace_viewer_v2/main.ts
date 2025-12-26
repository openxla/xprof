import type {WasmModule} from './trace_viewer_v2_wasm/trace_viewer_v2';

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
 * The name of the request data custom event, dispatched from the UI when the
 * user requests new data (e.g. by zooming or panning).
 */
export const FETCH_DATA_EVENT_NAME = 'fetch_data';

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

/**
 * Interface for the WebAssembly module loaded by `loadWasmTraceViewerModule`.
 * This interface extends the base `WasmModule` and includes properties and
 * methods specific to the Trace Viewer v2 WASM application. It defines the
 * API through which the JavaScript/TypeScript code interacts with the
 * compiled C++ Trace Viewer logic, including canvas access, WebGPU device
 * injection, and trace data processing.
 */
export declare interface TraceViewerV2Module extends WasmModule {
  HEAPU8: Uint8Array;
  canvas: HTMLCanvasElement;
  callMain(args: string[]): void;
  preinitializedWebGPUDevice: GPUDevice | null;
  processTraceEvents(
      data: TraceData,
      timeRangeFromUrl?: [number, number],
      ): void;
  loadJsonData?(url: string): Promise<void>;
  getProcessList?(url: string): Promise<string[] | undefined>;
  StringVector: {
    size(): number;
    get(index: number): string;
    toArray(): string[];
  };
  Application: {
    Instance(): {
      data_provider(): {
        getProcessList(): TraceViewerV2Module['StringVector'];
      };
    };
  };
}

declare interface TraceData {
  traceEvents: Array<{[key: string]: unknown}>;
  fullTimespan?: [number, number];
}

// Type guard to check if an object conforms to the TraceData interface
function isTraceData(data: unknown): data is TraceData {
  return (
    typeof data === 'object' &&
    data !== null &&
    data.hasOwnProperty('traceEvents') &&
    Array.isArray((data as TraceData).traceEvents)
  );
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
  void device.lost.then(() => {
    throw new Error('WebGPU Cannot be initialized - Device has been lost');
  });
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
  const traceviewerModule = await loadWasmTraceViewerModule(moduleConfig);
  traceviewerModule.preinitializedWebGPUDevice = device;
  traceviewerModule.callMain([]);
  return traceviewerModule;
}

async function initGpuAndStartWasmApp(): Promise<TraceViewerV2Module> {
  const canvas = document.querySelector('#canvas') as HTMLCanvasElement;
  if (!canvas) {
    throw new Error('Could not find canvas element with id="canvas"');
  }
  const device = await getWebGpuDevice();
  configureCanvas(canvas, device);
  return loadAndStartWasm(canvas, device);
}

function setupFileInputHandler(traceviewerModule: TraceViewerV2Module) {
  const fileInput = document.getElementById('fileInput') as HTMLInputElement;
  if (fileInput) {
    fileInput.addEventListener('change', async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (!file) {
        return;
      }

      try {
        const fileContent = await file.text();
        const jsonData = JSON.parse(fileContent);
        if (!isTraceData(jsonData)) {
          console.error('File does not contain valid trace events.');
          return;
        }
        traceviewerModule.processTraceEvents(jsonData, undefined);
      } catch (error) {
        console.error('Error processing file:', error);
      }
    });
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
    canvas: HTMLCanvasElement|null|undefined,
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

function getTimeRangeFromUrl(urlObj: URL): [number, number]|undefined {
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

function isFetchDataEvent(
    event: Event,
    ): event is CustomEvent<FetchDataEventDetail> {
  return (
      event instanceof CustomEvent && event.detail &&
      typeof event.detail.start_time_ms === 'number' &&
      typeof event.detail.end_time_ms === 'number');
}

async function handleFetchDataEvent(
    event: Event,
    currentDataUrl: string|null,
    traceviewerModule: TraceViewerV2Module|null,
) {
  if (!isFetchDataEvent(event)) {
    return;
  }
  const detail = event.detail;
  if (!currentDataUrl) {
    console.warn('Data URL not set, cannot fetch new data.');
    return;
  }
  if (!traceviewerModule) {
    console.warn('Trace viewer module not initialized.');
    return;
  }

  try {
    const urlObj = new URL(currentDataUrl, window.location.href);

    urlObj.searchParams.set(
        TRACE_VIEW_OPTION.START_TIME_MS, String(detail.start_time_ms));
    urlObj.searchParams.set(
        TRACE_VIEW_OPTION.END_TIME_MS, String(detail.end_time_ms));

    // Update resolution
    updateUrlWithResolution(urlObj, traceviewerModule.canvas);

    // TODO(b/470214911): Add support for additional query parameters to allow
    // for filtering by specific events, groups, or other criteria.
    const jsonData = await loadJsonDataInternal(urlObj.toString());
    if (!isTraceData(jsonData)) {
      console.error('File does not contain valid trace events.');
      return;
    }
    traceviewerModule.processTraceEvents(
        jsonData, /* timeRangeFromUrl= */ undefined);
  } catch (e) {
    console.error('Error fetching new data:', e);
  }
}

/**
 * Initializes the Trace Viewer v2 application.
 * This function sets up the necessary environment, including requesting a
 * WebGPU device, configuring a canvas for WebGPU rendering, and loading the
 * WebAssembly module for the trace viewer. It also exposes a method on the
 * returned module to load trace data from a JSON URL. This is the main entry
 * point for the Trace Viewer v2.
 *
 * @return A promise that resolves with the initialized TraceViewerV2Module, or
 *     null if initialization fails.
 */
export async function traceViewerV2Main(): Promise<TraceViewerV2Module|null> {
  let traceviewerModule: TraceViewerV2Module|null = null;
  let currentDataUrl: string|null = null;

  try {
    traceviewerModule = await initGpuAndStartWasmApp();
  } catch (e) {
    const error = e as Error;
    console.error('Application Initialization Failed:', error);
    return null;
  }

  setupFileInputHandler(traceviewerModule);

  // Add a method to the module to load data from a URL
  traceviewerModule.loadJsonData = async (url: string) => {
    currentDataUrl = url;
    try {
      window.dispatchEvent(new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {status: TraceViewerV2LoadingStatus.LOADING_DATA},
      }));

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

      updateUrlWithResolution(urlObj, traceviewerModule.canvas);

      const jsonData = await loadJsonDataInternal(urlObj.toString());

      if (!isTraceData(jsonData)) {
        console.error('File does not contain valid trace events.');

        window.dispatchEvent(new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
          detail: {status: TraceViewerV2LoadingStatus.IDLE},
        }));

        return;
      }

      window.dispatchEvent(new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {status: TraceViewerV2LoadingStatus.PROCESSING_DATA},
      }));

      // Yield to the event loop to allow the UI to re-render and display the
      // 'Processing data' status before the potentially long-running
      // processTraceEvents call.
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });

      traceviewerModule.processTraceEvents(jsonData, timeRangeFromUrl);

      window.dispatchEvent(new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {status: TraceViewerV2LoadingStatus.IDLE},
      }));
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
    }
  };

  window.addEventListener(FETCH_DATA_EVENT_NAME, (event: Event) => {
    handleFetchDataEvent(event, currentDataUrl, traceviewerModule);
  });

  // TODO(b/459575608): This should be updated when emscripten bindings
  // are updated.
  traceviewerModule.getProcessList = async (url: string) => {
    const processList = traceviewerModule.Application.Instance()
      .data_provider()
      .getProcessList();
    const processArray: string[] = [];
    if (
      typeof processList.size === 'function' &&
      typeof processList.get === 'function'
    ) {
      const size = processList.size();
      for (let i = 0; i < size; i++) {
        processArray.push(processList.get(i));
      }
    } else {
      console.error(
        'getProcessList result does not have size() or get() methods.',
      );
    }
    return processArray;
  };

  return traceviewerModule;
}
