import type {WasmModule} from './trace_viewer_v2_wasm/trace_viewer_v2';

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
 * This function sets up the necessary environment, including requesting a WebGPU device,
 * configuring a canvas for WebGPU rendering, and loading the WebAssembly module
 * for the trace viewer. It also exposes a method on the returned module to load
 * trace data from a JSON URL. This is the main entry point for the Trace Viewer v2.
 *
 * @return A promise that resolves with the initialized TraceViewerV2Module.
 */
export async function traceViewerV2Main(): Promise<TraceViewerV2Module | null> {
  let traceviewerModule: TraceViewerV2Module | null = null;

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
      const jsonData = await loadJsonDataInternal(url);
      if (!isTraceData(jsonData)) {
        console.error('File does not contain valid trace events.');
        return;
      }
      traceviewerModule.processTraceEvents(jsonData);
    } catch (e) {
      console.error('Error processing file:', e);
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
