import {NgModule} from '@angular/core';

import {PerfCounters} from './perf_counters';

/** A perf counters module. */
@NgModule({
  imports: [PerfCounters],
  exports: [PerfCounters],
})
export class PerfCountersModule {
}
