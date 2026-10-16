import {NgModule} from '@angular/core';
import {MaxHeapChart} from './max_heap_chart';

/** @deprecated Import the standalone MaxHeapChart component directly instead. */
@NgModule({
  imports: [MaxHeapChart],
  exports: [MaxHeapChart],
})
export class MaxHeapChartModule {}
