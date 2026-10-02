import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';

import {InferenceLatencyChart} from './inference_latency_chart';

@NgModule({
  imports: [MatCardModule, InferenceLatencyChart],
  exports: [InferenceLatencyChart],
})
export class InferenceLatencyChartModule {}
