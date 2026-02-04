import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';

import {TraceViewer} from './trace_viewer';
import {TraceViewerContainer} from './trace_viewer_container';

/** A trace viewer module. */
@NgModule({
  declarations: [TraceViewer, TraceViewerContainer],
  imports: [
    CommonModule,
    MatButtonModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatProgressBarModule,
    MatProgressSpinnerModule,
    PipesModule,
  ],
  exports: [TraceViewer, TraceViewerContainer]
})
export class TraceViewerModule {
}
