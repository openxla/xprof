import {NgModule} from '@angular/core';

import {PodViewerDetails} from './pod_viewer_details';

/** A pod viewer details view module. */
@NgModule({
  imports: [
    PodViewerDetails,
  ],
  exports: [PodViewerDetails]
})
export class PodViewerDetailsModule {
}
