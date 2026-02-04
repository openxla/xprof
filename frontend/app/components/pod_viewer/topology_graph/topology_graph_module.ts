import {NgModule} from '@angular/core';
import {TopologyGraph} from './topology_graph';

/** A topology graph view module. */
@NgModule({
  imports: [TopologyGraph],
  exports: [TopologyGraph]
})
export class TopologyGraphModule {
}
