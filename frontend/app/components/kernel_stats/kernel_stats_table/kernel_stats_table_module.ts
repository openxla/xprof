import {NgModule} from '@angular/core';

import {KernelStatsTable} from './kernel_stats_table';

@NgModule({
  imports: [KernelStatsTable],
  exports: [KernelStatsTable]
})
export class KernelStatsTableModule {
}
