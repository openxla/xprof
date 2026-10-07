import {NgModule} from '@angular/core';
import {InferenceLatencyChart} from './inference_latency_chart';

/** @deprecated Import the standalone InferenceLatencyChart component directly instead. */
@NgModule({
  imports: [InferenceLatencyChart],
  exports: [InferenceLatencyChart],
})
export class InferenceLatencyChartModule {}
