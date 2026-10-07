import {NgModule} from '@angular/core';
import {PodViewer} from './pod_viewer';

/** @deprecated Import the standalone PodViewer component directly instead. */
@NgModule({
  imports: [PodViewer],
  exports: [PodViewer],
})
export class PodViewerModule {}
