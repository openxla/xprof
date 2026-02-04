import {NgModule} from '@angular/core';

import {PodViewer} from './pod_viewer';

/** A pod viewer module. */
@NgModule({
  imports: [PodViewer],
  exports: [PodViewer]
})
export class PodViewerModule {
}
