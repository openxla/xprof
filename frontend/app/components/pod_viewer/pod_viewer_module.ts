import {NgModule} from '@angular/core';
import {PodViewer} from './pod_viewer';

@NgModule({
  imports: [PodViewer],
  exports: [PodViewer],
})
export class PodViewerModule {}
