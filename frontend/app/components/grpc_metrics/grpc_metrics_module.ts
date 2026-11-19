import {NgModule} from '@angular/core';
import {GrpcMetrics} from './grpc_metrics';

@NgModule({
  imports: [GrpcMetrics],
  exports: [GrpcMetrics],
})
export class GrpcMetricsModule {}
