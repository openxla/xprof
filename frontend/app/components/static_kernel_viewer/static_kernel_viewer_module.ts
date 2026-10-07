import {NgModule} from '@angular/core';
import {StaticKernelViewer} from './static_kernel_viewer';

/** @deprecated Import the standalone StaticKernelViewer component directly instead. */
@NgModule({
  imports: [StaticKernelViewer],
  exports: [StaticKernelViewer],
})
export class StaticKernelViewerModule {}
