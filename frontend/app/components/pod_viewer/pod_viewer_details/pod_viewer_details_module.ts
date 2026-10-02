import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';

import {PodViewerDetails} from './pod_viewer_details';

/** A pod viewer details view module. */
@NgModule({
  imports: [CommonModule, MatCardModule, PodViewerDetails],
  exports: [PodViewerDetails],
})
export class PodViewerDetailsModule {}
