import {NgModule} from '@angular/core';
import {MatDividerModule} from '@angular/material/divider';
import {ExportAsCsvModule} from 'org_xprof/frontend/app/components/controls/export_as_csv/export_as_csv_module';
import {KernelStatsTableModule} from 'org_xprof/frontend/app/components/kernel_stats/kernel_stats_table/kernel_stats_table_module';

import {KernelStats} from './kernel_stats';

/** A kernel stats module. */
@NgModule({
  imports: [
    MatDividerModule,
    ExportAsCsvModule,
    KernelStatsTableModule,
    KernelStats,
  ],
  exports: [KernelStats],
})
export class KernelStatsModule {}
