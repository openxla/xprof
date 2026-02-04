import {NgModule} from '@angular/core';


import {KernelStats} from './kernel_stats';

/** A kernel stats module. */
@NgModule({
  imports: [KernelStats],
  exports: [KernelStats]
})
export class KernelStatsModule {
}
