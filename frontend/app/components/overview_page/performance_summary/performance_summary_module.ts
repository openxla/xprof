import {NgModule} from '@angular/core';
import {PerformanceSummary} from './performance_summary';

/** @deprecated Import the standalone PerformanceSummary component directly instead. */
@NgModule({
  imports: [PerformanceSummary],
  exports: [PerformanceSummary],
})
export class PerformanceSummaryModule {}
