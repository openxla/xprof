import {NgModule} from '@angular/core';
import {ProgramLevelAnalysis} from './program_level_analysis';

/** @deprecated Import the standalone ProgramLevelAnalysis component directly instead. */
@NgModule({
  imports: [ProgramLevelAnalysis],
  exports: [ProgramLevelAnalysis],
})
export class ProgramLevelAnalysisModule {}
