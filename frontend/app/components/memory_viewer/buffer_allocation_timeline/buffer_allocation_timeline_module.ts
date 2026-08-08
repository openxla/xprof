import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';
import {BufferAllocationTimeline} from './buffer_allocation_timeline';

@NgModule({
  declarations: [BufferAllocationTimeline],
  imports: [CommonModule, MatIconModule, MatTooltipModule],
  exports: [BufferAllocationTimeline],
})
export class BufferAllocationTimelineModule {}
