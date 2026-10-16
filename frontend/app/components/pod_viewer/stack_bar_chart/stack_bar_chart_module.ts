import {NgModule} from '@angular/core';
import {StackBarChart} from './stack_bar_chart';

/** @deprecated Import the standalone StackBarChart component directly instead. */
@NgModule({
  imports: [StackBarChart],
  exports: [StackBarChart],
})
export class StackBarChartModule {}
