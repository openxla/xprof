import {NgModule} from '@angular/core';

import {ProgramLevelAnalysis} from './program_level_analysis';

@NgModule({
  imports: [ProgramLevelAnalysis],
  exports: [ProgramLevelAnalysis],
})
export class ProgramLevelAnalysisModule {}
