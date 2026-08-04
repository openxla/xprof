import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {BufferAllocationTimeline} from './buffer_allocation_timeline';

@NgModule({
  declarations: [BufferAllocationTimeline],
  imports: [CommonModule, MatIconModule],
  exports: [BufferAllocationTimeline],
})
export class BufferAllocationTimelineModule {}
