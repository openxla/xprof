import {NgModule} from '@angular/core';
import {KernelStats} from './kernel_stats';

/** @deprecated Import the standalone KernelStats component directly instead. */
@NgModule({
  imports: [KernelStats],
  exports: [KernelStats],
})
export class KernelStatsModule {}
