import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {TraceViewerContainer} from 'org_xprof/frontend/app/components/trace_viewer_container/trace_viewer_container';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';

import {TraceViewer} from './trace_viewer';

/** A trace viewer module. */
@NgModule({
  declarations: [TraceViewer],
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    PipesModule,
    TraceViewerContainer,
  ],
  providers: [DataServiceV2],
  exports: [TraceViewer]
})
export class TraceViewerModule {
}
