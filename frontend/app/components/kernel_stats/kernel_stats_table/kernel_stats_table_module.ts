import {NgModule} from '@angular/core';
import {KernelStatsTable} from './kernel_stats_table';

/** @deprecated Import the standalone KernelStatsTable component directly instead. */
@NgModule({
  imports: [KernelStatsTable],
  exports: [KernelStatsTable],
})
export class KernelStatsTableModule {}
