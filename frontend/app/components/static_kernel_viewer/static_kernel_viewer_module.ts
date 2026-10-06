import {NgModule} from '@angular/core';
import {StaticKernelViewer} from './static_kernel_viewer';

@NgModule({
  imports: [StaticKernelViewer],
  exports: [StaticKernelViewer],
})
export class StaticKernelViewerModule {}
