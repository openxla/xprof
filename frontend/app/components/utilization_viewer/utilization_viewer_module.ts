import {NgModule} from '@angular/core';
import {UtilizationViewer} from './utilization_viewer';

/** @deprecated Import the standalone UtilizationViewer component directly instead. */
@NgModule({
  imports: [UtilizationViewer],
  exports: [UtilizationViewer],
})
export class UtilizationViewerModule {}
