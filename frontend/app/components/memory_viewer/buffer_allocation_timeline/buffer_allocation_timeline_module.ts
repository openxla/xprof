import {NgModule} from '@angular/core';
import {BufferAllocationTimeline} from './buffer_allocation_timeline';

@NgModule({
  imports: [BufferAllocationTimeline],
  exports: [BufferAllocationTimeline],
})
export class BufferAllocationTimelineModule {}
