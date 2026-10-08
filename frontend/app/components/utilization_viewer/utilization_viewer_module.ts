import {NgModule} from '@angular/core';
import {UtilizationViewer} from './utilization_viewer';

@NgModule({
  imports: [UtilizationViewer],
  exports: [UtilizationViewer],
})
export class UtilizationViewerModule {}
