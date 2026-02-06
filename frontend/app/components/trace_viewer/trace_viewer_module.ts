import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';

import {TraceViewer} from './trace_viewer';

/** A trace viewer module. */
@NgModule({
  declarations: [TraceViewer],
  imports: [
    CommonModule,
    MatIconModule,
    MatProgressBarModule,
    PipesModule,
  ],
  exports: [TraceViewer]
})
export class TraceViewerModule {
}
