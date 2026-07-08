import {TraceViewerV2Module} from '../components/trace_viewer_v2/main';

/** Mock memory pointer returned by simulated malloc calls in WASM heap. */
export const MOCK_MALLOC_PTR = 12345;

/** Interface defining a dictionary of Jasmine spies for WASM module mocking. */
export interface MockSpies {
  [key: string]: jasmine.Spy | undefined;
}

/**
 * Mocks the loading of the Trace Viewer v2 WASM module by injecting spies and
 * protecting immutable properties on the window object.
 * @param customSpies Dictionary of custom Jasmine spies to inject into the mock.
 */
export function mockWasmModuleLoad(customSpies: MockSpies): void {
  const mockModule = {
    setSearchResultsInWasm:
      customSpies['setSearchResultsInWasm'] ||
      jasmine.createSpy('setSearchResultsInWasm'),
    setCompressedSearchResultsInWasm:
      customSpies['setCompressedSearchResultsInWasm'] ||
      jasmine.createSpy('setCompressedSearchResultsInWasm'),
    processCompressedTraceEvents:
      customSpies['processCompressedTraceEvents'] ||
      jasmine.createSpy('processCompressedTraceEvents'),
    processTraceEvents:
      customSpies['processTraceEvents'] ||
      jasmine.createSpy('processTraceEvents'),
    processPerfettoTraceEvents:
      customSpies['processPerfettoTraceEvents'] ||
      jasmine.createSpy('processPerfettoTraceEvents'),
    _malloc:
      customSpies['_malloc'] ||
      jasmine.createSpy('_malloc').and.returnValue(MOCK_MALLOC_PTR),
    _free: customSpies['_free'] || jasmine.createSpy('_free'),
    SetPalette: customSpies['SetPalette'] || jasmine.createSpy('SetPalette'),
    canvas: document.createElement('canvas'),
    preinitializedWebGPUDevice: null,
    getAllFlowCategories:
      customSpies['getAllFlowCategories'] ||
      jasmine.createSpy('getAllFlowCategories').and.returnValue([]),
    HEAPU8: new Uint8Array(65536),
    StringVector: {
      size: () => 0,
      get: () => '',
      toArray: () => [],
    },
    IntVector: {
      size: () => 0,
      get: () => 0,
    },
    application: {
      instance: () => ({
        shutdown: customSpies['shutdown'] || jasmine.createSpy('shutdown'),
        dataProvider: () => ({
          getFlowCategories: () => ({size: () => 0, get: () => 0}),
          getProcessMappings: () => ({}),
          getProcessNames: () => ({}),
        }),
        getCurrentSearchResultIndex: () => 0,
        getSearchResultsCount: () => 0,
        navigateToNextSearchResult:
          customSpies['navigateToNextSearchResult'] ||
          jasmine.createSpy('navigateToNextSearchResult'),
        navigateToPrevSearchResult:
          customSpies['navigateToPrevSearchResult'] ||
          jasmine.createSpy('navigateToPrevSearchResult'),
        resize: customSpies['resize'] || jasmine.createSpy('resize'),
        setSearchQuery:
          customSpies['setSearchQuery'] || jasmine.createSpy('setSearchQuery'),
        setMouseMode:
          customSpies['setMouseMode'] || jasmine.createSpy('setMouseMode'),
        setVisibleFlowCategory:
          customSpies['setVisibleFlowCategory'] ||
          jasmine.createSpy('setVisibleFlowCategory'),
        setVisibleFlowCategories:
          customSpies['setVisibleFlowCategories'] ||
          jasmine.createSpy('setVisibleFlowCategories'),
      }),
    },
    getFeatureFlag:
      customSpies['getFeatureFlag'] ||
      jasmine.createSpy('getFeatureFlag').and.returnValue(false),
    callMain: customSpies['callMain'] || jasmine.createSpy('callMain'),
  };

  const protectedModule: TraceViewerV2Module = {} as TraceViewerV2Module;
  for (const key of Object.keys(mockModule)) {
    Object.defineProperty(protectedModule, key, {
      get: () => mockModule[key as keyof typeof mockModule],
      set: () => {}, // Immutable mock protection: no-op setter
      configurable: true,
      enumerable: true,
    });
  }

  Object.defineProperty(protectedModule, 'loadTraceData', {
    get: () =>
      customSpies['loadTraceData'] ||
      jasmine.createSpy('loadTraceData').and.resolveTo(),
    set: () => {},
    configurable: true,
  });

  Object.defineProperty(protectedModule, 'loadSearchResults', {
    get: () =>
      customSpies['loadSearchResults'] ||
      jasmine.createSpy('loadSearchResults').and.resolveTo(),
    set: () => {},
    configurable: true,
  });

  Object.defineProperty(window, 'loadWasmTraceViewerModule', {
    get: () => () => Promise.resolve(protectedModule),
    set: () => {}, // Immutable mock protection: no-op setter
    configurable: true,
    enumerable: true,
  });
}

/**
 * Sets up a mock WebGPU canvas context on the provided canvas element.
 * @param canvas The HTML canvas element to mock.
 */
export function setupWebGpuCanvasMock(canvas: HTMLCanvasElement): void {
  const mockContext = jasmine.createSpyObj('GPUCanvasContext', ['configure']);
  spyOn(canvas, 'getContext').and.returnValue(mockContext);
}

/**
 * Creates a mock File object with the specified content, name, and MIME type.
 * @param content The string content of the file.
 * @param name The name of the file.
 * @param type The MIME type of the file.
 * @return A mock File object with arrayBuffer implemented.
 */
export function createMockFile(
  content: string,
  name: string,
  type = 'application/octet-stream',
): File {
  const file = new File([content], name, {type});
  file.arrayBuffer = () =>
    Promise.resolve(new TextEncoder().encode(content).buffer);
  return file;
}
