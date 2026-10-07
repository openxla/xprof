import {NgModule} from '@angular/core';
import {FlopRateChart} from './flop_rate_chart';

/** @deprecated Import the standalone FlopRateChart component directly instead. */
@NgModule({
  imports: [FlopRateChart],
  exports: [FlopRateChart],
})
export class FlopRateChartModule {}
