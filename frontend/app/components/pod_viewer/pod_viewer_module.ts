import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatDividerModule} from '@angular/material/divider';
import {MatLegacySliderModule} from '@angular/material/slider';

import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';
import {PodViewer} from './pod_viewer';
import {StackBarChartModule} from './stack_bar_chart/stack_bar_chart_module';
import {TopologyGraphModule} from './topology_graph/topology_graph_module';

/** A pod viewer module. */
@NgModule({
  declarations: [PodViewer],
  imports: [
    CommonModule,
    DiagnosticsViewModule,
    MatDividerModule,
    MatLegacySliderModule,
    StackBarChartModule,
    TopologyGraphModule,
  ],
  exports: [PodViewer]
})
export class PodViewerModule {
}
