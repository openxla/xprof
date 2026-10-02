import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDividerModule} from '@angular/material/divider';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {ExportAsCsvModule} from 'org_xprof/frontend/app/components/controls/export_as_csv/export_as_csv_module';
import {FlopRateChartModule} from 'org_xprof/frontend/app/components/framework_op_stats/flop_rate_chart/flop_rate_chart_module';
import {ModelPropertiesModule} from 'org_xprof/frontend/app/components/framework_op_stats/model_properties/model_properties_module';
import {OperationsTableModule} from 'org_xprof/frontend/app/components/framework_op_stats/operations_table/operations_table_module';
import {StatsTableModule} from 'org_xprof/frontend/app/components/framework_op_stats/stats_table/stats_table_module';

import {FrameworkOpStats} from './framework_op_stats';

/** An op profile module. */
@NgModule({
  imports: [
    CommonModule,
    MatButtonModule,
    MatDividerModule,
    MatIconModule,
    MatMenuModule,
    ChartModule,
    ExportAsCsvModule,
    FlopRateChartModule,
    ModelPropertiesModule,
    OperationsTableModule,
    StatsTableModule,
    FrameworkOpStats,
  ],
  exports: [FrameworkOpStats],
})
export class FrameworkOpStatsModule {}
