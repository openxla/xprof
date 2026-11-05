import {NgModule} from '@angular/core';
import {DiagnosticsView} from './diagnostics_view';

/** @deprecated Import the standalone DiagnosticsView component directly instead. */
@NgModule({
  imports: [DiagnosticsView],
  exports: [DiagnosticsView],
})
export class DiagnosticsViewModule {}
