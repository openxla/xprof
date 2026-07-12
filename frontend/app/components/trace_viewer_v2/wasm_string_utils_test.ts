import {
  freeWasmPtr,
  WasmHeapModule,
  withWasmCString,
  withWasmHeapBuffer,
  withWasmMalloc,
} from './wasm_string_utils';

interface MockWasmModule extends WasmHeapModule {
  freeCount: number;
  lastFreedPtr: number | null;
}

/**
 * Lightweight mock of the Emscripten heap surface used by wasm_string_utils.
 * Tracks free invocations so tests can assert leak-free ownership.
 */
function createMockModule(options?: {
  failMalloc?: boolean;
  heapSize?: number;
}): MockWasmModule {
  const heapSize = options?.heapSize ?? 4096;
  const heap = new Uint8Array(heapSize);
  let nextPtr = 64;
  const mock: MockWasmModule = {
    HEAPU8: heap,
    freeCount: 0,
    lastFreedPtr: null,
    _malloc(size: number): number {
      if (options?.failMalloc) {
        return 0;
      }
      const ptr = nextPtr;
      nextPtr += size + 8;
      if (nextPtr > heapSize) {
        throw new Error('mock heap exhausted');
      }
      return ptr;
    },
    _free(ptr: number): void {
      mock.freeCount += 1;
      mock.lastFreedPtr = ptr;
    },
  };
  return mock;
}

describe('wasm_string_utils', () => {
  describe('freeWasmPtr', () => {
    it('no-ops for null, undefined, and 0', () => {
      const module = createMockModule();
      freeWasmPtr(module, null);
      freeWasmPtr(module, undefined);
      freeWasmPtr(module, 0);
      expect(module.freeCount).toBe(0);
    });

    it('frees non-zero pointers', () => {
      const module = createMockModule();
      freeWasmPtr(module, 42);
      expect(module.freeCount).toBe(1);
      expect(module.lastFreedPtr).toBe(42);
    });
  });

  describe('withWasmMalloc', () => {
    it('frees after a successful callback', () => {
      const module = createMockModule();
      const result = withWasmMalloc(module, 16, (ptr) => {
        expect(ptr).toBeGreaterThan(0);
        return 'ok';
      });
      expect(result).toBe('ok');
      expect(module.freeCount).toBe(1);
    });

    it('frees even when the callback throws', () => {
      const module = createMockModule();
      expect(() =>
        withWasmMalloc(module, 16, () => {
          throw new Error('callback boom');
        }),
      ).toThrowError('callback boom');
      expect(module.freeCount).toBe(1);
    });

    it('does not call free when malloc fails', () => {
      const module = createMockModule({failMalloc: true});
      expect(() => withWasmMalloc(module, 16, () => 'x')).toThrowError(
        /Failed to allocate WASM memory buffer/,
      );
      expect(module.freeCount).toBe(0);
    });

    it('skips malloc/free for zero-length allocations', () => {
      const module = createMockModule();
      const result = withWasmMalloc(module, 0, (ptr) => {
        expect(ptr).toBe(0);
        return 123;
      });
      expect(result).toBe(123);
      expect(module.freeCount).toBe(0);
    });
  });

  describe('withWasmHeapBuffer', () => {
    it('copies bytes and always frees', () => {
      const module = createMockModule();
      const data = new Uint8Array([1, 2, 3, 4]);
      const seen: number[] = [];
      withWasmHeapBuffer(module, data, (ptr, size) => {
        expect(size).toBe(4);
        for (let i = 0; i < size; i++) {
          seen.push(module.HEAPU8[ptr + i]);
        }
      });
      expect(seen).toEqual([1, 2, 3, 4]);
      expect(module.freeCount).toBe(1);
    });

    it('frees when the native-style callback throws', () => {
      const module = createMockModule();
      expect(() =>
        withWasmHeapBuffer(module, new Uint8Array([9]), () => {
          throw new Error('native failed');
        }),
      ).toThrowError('native failed');
      expect(module.freeCount).toBe(1);
    });
  });

  describe('withWasmCString', () => {
    it('writes a null-terminated UTF-8 string and frees', () => {
      const module = createMockModule();
      withWasmCString(module, 'hi', (ptr) => {
        expect(module.HEAPU8[ptr]).toBe('h'.charCodeAt(0));
        expect(module.HEAPU8[ptr + 1]).toBe('i'.charCodeAt(0));
        expect(module.HEAPU8[ptr + 2]).toBe(0);
      });
      expect(module.freeCount).toBe(1);
    });

    it('frees when the callback throws', () => {
      const module = createMockModule();
      expect(() =>
        withWasmCString(module, 'leak-me', () => {
          throw new Error('cstr boom');
        }),
      ).toThrowError('cstr boom');
      expect(module.freeCount).toBe(1);
    });
  });
});
