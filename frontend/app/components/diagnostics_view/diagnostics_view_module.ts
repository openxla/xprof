import {NgModule} from '@angular/core';
import {DiagnosticsView} from './diagnostics_view';

/**
 * NgModule shim for backwards compatibility with non-standalone callers.
 */
@NgModule({
  imports: [DiagnosticsView],
  exports: [DiagnosticsView],
})
export class DiagnosticsViewModule {}
