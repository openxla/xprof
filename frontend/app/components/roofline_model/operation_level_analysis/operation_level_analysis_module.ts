import {NgModule} from '@angular/core';
import {OperationLevelAnalysis} from './operation_level_analysis';

/** @deprecated Import the standalone OperationLevelAnalysis component directly instead. */
@NgModule({
  imports: [OperationLevelAnalysis],
  exports: [OperationLevelAnalysis],
})
export class OperationLevelAnalysisModule {}
