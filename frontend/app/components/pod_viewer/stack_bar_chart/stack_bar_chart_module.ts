import {NgModule} from '@angular/core';

import {StackBarChart} from './stack_bar_chart';

/** A stack bar chart view module. */
@NgModule({
  imports: [StackBarChart],
  exports: [StackBarChart],
})
export class StackBarChartModule {
}
