import {NgModule} from '@angular/core';
import {AnalysisSummary} from './analysis_summary';

/** @deprecated Import the standalone AnalysisSummary component directly instead. */
@NgModule({
  imports: [AnalysisSummary],
  exports: [AnalysisSummary],
})
export class AnalysisSummaryModule {}
