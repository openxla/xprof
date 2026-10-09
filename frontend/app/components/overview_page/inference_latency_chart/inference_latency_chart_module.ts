import {NgModule} from '@angular/core';
import {InferenceLatencyChart} from './inference_latency_chart';

@NgModule({
  imports: [InferenceLatencyChart],
  exports: [InferenceLatencyChart],
})
export class InferenceLatencyChartModule {}
