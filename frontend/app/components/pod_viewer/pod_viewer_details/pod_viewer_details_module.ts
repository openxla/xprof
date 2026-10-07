import {NgModule} from '@angular/core';
import {PodViewerDetails} from './pod_viewer_details';

/** @deprecated Import the standalone PodViewerDetails component directly instead. */
@NgModule({
  imports: [PodViewerDetails],
  exports: [PodViewerDetails],
})
export class PodViewerDetailsModule {}
