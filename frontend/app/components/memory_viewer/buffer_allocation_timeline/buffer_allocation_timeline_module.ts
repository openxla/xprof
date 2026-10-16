import {NgModule} from '@angular/core';
import {BufferAllocationTimeline} from './buffer_allocation_timeline';

/** @deprecated Import the standalone BufferAllocationTimeline component directly instead. */
@NgModule({
  imports: [BufferAllocationTimeline],
  exports: [BufferAllocationTimeline],
})
export class BufferAllocationTimelineModule {}
