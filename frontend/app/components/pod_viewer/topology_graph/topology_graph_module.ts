import {NgModule} from '@angular/core';
import {TopologyGraph} from './topology_graph';

/** @deprecated Import the standalone TopologyGraph component directly instead. */
@NgModule({
  imports: [TopologyGraph],
  exports: [TopologyGraph],
})
export class TopologyGraphModule {}
