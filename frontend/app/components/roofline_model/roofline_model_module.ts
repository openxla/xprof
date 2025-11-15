import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatTooltipModule} from '@angular/material/tooltip';
import {TableModule} from 'org_xprof/frontend/app/components/chart/table/table_module';
import {CategoryFilterModule} from 'org_xprof/frontend/app/components/controls/category_filter/category_filter_module';
import {ExportAsCsvModule} from 'org_xprof/frontend/app/components/controls/export_as_csv/export_as_csv_module';
import {StringFilterModule} from 'org_xprof/frontend/app/components/controls/string_filter/string_filter_module';
import {OperationLevelAnalysisModule} from 'org_xprof/frontend/app/components/roofline_model/operation_level_analysis/operation_level_analysis_module';
import {ProgramLevelAnalysisModule} from 'org_xprof/frontend/app/components/roofline_model/program_level_analysis/program_level_analysis_module';

import {RooflineModel} from './roofline_model';

/** A roofline model module. */
@NgModule({
  declarations: [RooflineModel],
  imports: [
    CommonModule,
    TableModule,
    ExportAsCsvModule,
    CategoryFilterModule,
    StringFilterModule,
    ProgramLevelAnalysisModule,
    OperationLevelAnalysisModule,
    MatTooltipModule,
    MatSlideToggleModule,
    MatIconModule,
    MatProgressBarModule,
  ],
  exports: [RooflineModel],
})
export class RooflineModelModule {
}
