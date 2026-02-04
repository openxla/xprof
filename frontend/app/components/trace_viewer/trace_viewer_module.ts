import {NgModule} from '@angular/core';

import {TraceViewer} from './trace_viewer';

/** A trace viewer module. */
@NgModule({
  imports: [TraceViewer],
  exports: [TraceViewer]
})
export class TraceViewerModule {
}
