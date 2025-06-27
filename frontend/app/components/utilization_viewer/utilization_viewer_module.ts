import {NgModule} from '@angular/core';

import {UtilizationViewer} from './utilization_viewer';

/** Utilization viewer module. */
@NgModule({
  imports: [UtilizationViewer],
  exports: [UtilizationViewer],
})
export class UtilizationViewerModule {
}
