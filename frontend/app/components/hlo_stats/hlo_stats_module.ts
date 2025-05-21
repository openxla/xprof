import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FormsModule, ReactiveFormsModule} from '@angular/forms';
import {MatDividerModule} from '@angular/material/divider';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatSelectModule} from '@angular/material/select';
import {MatTooltipModule} from '@angular/material/tooltip';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {CategoryFilterModule} from 'org_xprof/frontend/app/components/controls/category_filter/category_filter_module';
import {ExportAsCsvModule} from 'org_xprof/frontend/app/components/controls/export_as_csv/export_as_csv_module';
import {StringFilterModule} from 'org_xprof/frontend/app/components/controls/string_filter/string_filter_module';
import {FlopRateChartModule} from 'org_xprof/frontend/app/components/framework_op_stats/flop_rate_chart/flop_rate_chart_module';

import {HloStats} from './hlo_stats';

/** An HLO stats module. */
@NgModule({
  declarations: [HloStats],
  imports: [
    ChartModule,
    CommonModule,
    FlopRateChartModule,
    CategoryFilterModule,
    ExportAsCsvModule,
    StringFilterModule,
    MatDividerModule,
    MatSelectModule,
    MatExpansionModule,
    MatTooltipModule,
    MatFormFieldModule,
    FormsModule,
    ReactiveFormsModule,
  ],
  exports: [HloStats],
})
export class HloStatsModule {
}
