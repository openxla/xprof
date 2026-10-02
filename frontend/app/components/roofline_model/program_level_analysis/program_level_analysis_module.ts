import {NgModule} from '@angular/core';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {TableModule} from 'org_xprof/frontend/app/components/chart/table/table_module';
import {CategoryFilterModule} from 'org_xprof/frontend/app/components/controls/category_filter/category_filter_module';

import {ProgramLevelAnalysis} from './program_level_analysis';

@NgModule({
  imports: [
    CategoryFilterModule,
    TableModule,
    ChartModule,
    ProgramLevelAnalysis,
  ],
  exports: [ProgramLevelAnalysis],
})
export class ProgramLevelAnalysisModule {}
