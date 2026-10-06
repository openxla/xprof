import {NgModule} from '@angular/core';
import {PerfCounters} from './perf_counters';

@NgModule({
  imports: [PerfCounters],
  exports: [PerfCounters],
})
export class PerfCountersModule {}
