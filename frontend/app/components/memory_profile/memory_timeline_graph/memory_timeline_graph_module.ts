import {NgModule} from '@angular/core';
import {MemoryTimelineGraph} from './memory_timeline_graph';

/** @deprecated Import the standalone MemoryTimelineGraph component directly instead. */
@NgModule({
  imports: [MemoryTimelineGraph],
  exports: [MemoryTimelineGraph],
})
export class MemoryTimelineGraphModule {}
