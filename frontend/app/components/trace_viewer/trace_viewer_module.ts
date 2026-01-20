import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';

import {TraceViewer} from './trace_viewer';
import {TraceViewerContainer} from './trace_viewer_container';

/** A trace viewer module. */
@NgModule({
  declarations: [TraceViewer, TraceViewerContainer],
  imports: [
    CommonModule,
    MatIconModule,
    MatProgressBarModule,
    PipesModule,
  ],
  exports: [TraceViewer, TraceViewerContainer]
})
export class TraceViewerModule {
}
