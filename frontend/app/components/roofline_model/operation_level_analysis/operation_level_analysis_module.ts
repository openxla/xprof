import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {TableModule} from 'org_xprof/frontend/app/components/chart/table/table_module';
import {CategoryFilterModule} from 'org_xprof/frontend/app/components/controls/category_filter/category_filter_module';
import {StringFilterModule} from 'org_xprof/frontend/app/components/controls/string_filter/string_filter_module';
import {StackTraceSnippetModule} from 'org_xprof/frontend/app/components/stack_trace_snippet/stack_trace_snippet_module';

import {OperationLevelAnalysis} from './operation_level_analysis';

@NgModule({
  declarations: [OperationLevelAnalysis],
  imports: [
    CategoryFilterModule,
    TableModule,
    StringFilterModule,
    ChartModule,
    MatIconModule,
    MatSlideToggleModule,
    StackTraceSnippetModule,
    CommonModule,
  ],
  exports: [OperationLevelAnalysis],
})
export class OperationLevelAnalysisModule {
}
