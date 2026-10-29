import {InjectionToken} from '@angular/core';
import {KernelViewerCache} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_viewer_cache';
import {IndexedDbKernelViewerStore} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_viewer_store';

/**
 * Provides the cache of kernel lists and traces of the Static Kernel Viewer,
 * by default a `KernelViewerCache` in IndexedDB that all viewers share.
 */
export const KERNEL_VIEWER_CACHE_TOKEN = new InjectionToken<KernelViewerCache>(
  'KernelViewerCache',
  {
    providedIn: 'root',
    factory: () =>
      new KernelViewerCache({store: new IndexedDbKernelViewerStore()}),
  },
);
