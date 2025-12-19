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
} as const;

/**
 * The name of the loading status update custom event, dispatched from WASM in
 * Trace Viewer v2.
 */
export const LOADING_STATUS_UPDATE_EVENT_NAME = 'loadingstatusupdate';

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
  processTraceEvents(data: TraceData): void;
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
        traceviewerModule.processTraceEvents(jsonData);
      } catch (error) {
        console.error('Error processing file:', error);
      }
    });
  }
}

/**
 * Updates a URL with a `resolution` parameter based on the canvas width.
 *
 * The resolution is calculated to optimize the number of trace events fetched
 * from the backend, preventing over-fetching of data that would not be visible.
 * If certain trace options are present (like filtering), resolution is set to 0
 * to fetch all data.
 *
 * @param url The URL to update.
 * @param canvas The canvas element used to determine the viewer width.
 * @return The updated URL string with the resolution parameter.
 */
export function updateUrlWithResolution(
    url: string,
    canvas: HTMLCanvasElement|null|undefined,
    ): string {
  let urlObj: URL;
  try {
    urlObj = new URL(url, window.location.href);
  } catch (e) {
    console.error('Invalid URL:', url, e);
    return url;
  }

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
  return urlObj.toString();
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
    try {
      window.dispatchEvent(new CustomEvent(LOADING_STATUS_UPDATE_EVENT_NAME, {
        detail: {status: TraceViewerV2LoadingStatus.LOADING_DATA},
      }));

      const fullUrl = updateUrlWithResolution(url, traceviewerModule.canvas);
      const jsonData = await loadJsonDataInternal(fullUrl);
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
      await new Promise(resolve => {
        setTimeout(resolve, 0);
      });

      traceviewerModule.processTraceEvents(jsonData);

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
