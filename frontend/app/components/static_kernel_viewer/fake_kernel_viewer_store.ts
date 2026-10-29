import {type KernelViewerStore} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_viewer_store';

/**
 * An in-memory `KernelViewerStore` for tests. It never expires or evicts
 * entries.
 */
export class FakeKernelViewerStore implements KernelViewerStore {
  readonly lists = new Map<string, unknown>();
  readonly traces = new Map<string, Blob>();

  readList(profileKey: string): Promise<unknown> {
    return Promise.resolve(this.lists.get(profileKey));
  }

  writeList(profileKey: string, response: unknown): Promise<void> {
    this.lists.set(profileKey, response);
    return Promise.resolve();
  }

  readTrace(traceKey: string): Promise<Blob | undefined> {
    return Promise.resolve(this.traces.get(traceKey));
  }

  writeTrace(traceKey: string, data: Blob): Promise<void> {
    this.traces.set(traceKey, data);
    return Promise.resolve();
  }
}
