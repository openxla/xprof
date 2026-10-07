import {NgModule} from '@angular/core';
import {PerfCounters} from './perf_counters';

/**
 * @deprecated Import the standalone `PerfCounters` component directly instead.
 */
@NgModule({
  imports: [PerfCounters],
  exports: [PerfCounters],
})
export class PerfCountersModule {}
