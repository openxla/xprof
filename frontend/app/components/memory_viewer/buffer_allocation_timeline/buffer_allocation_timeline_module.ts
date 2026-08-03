import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {BufferAllocationTimeline} from './buffer_allocation_timeline';

@NgModule({
  declarations: [BufferAllocationTimeline],
  imports: [CommonModule],
  exports: [BufferAllocationTimeline],
})
export class BufferAllocationTimelineModule {}
