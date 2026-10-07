import {NgModule} from '@angular/core';
import {ProgramOrderChart} from './program_order_chart';

/** @deprecated Import the standalone ProgramOrderChart component directly instead. */
@NgModule({
  imports: [ProgramOrderChart],
  exports: [ProgramOrderChart],
})
export class ProgramOrderChartModule {}
