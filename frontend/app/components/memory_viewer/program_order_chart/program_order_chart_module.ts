import {NgModule} from '@angular/core';


import {ProgramOrderChart} from './program_order_chart';

@NgModule({
  imports: [
    ProgramOrderChart,
  ],
  exports: [ProgramOrderChart]
})
export class ProgramOrderChartModule {
}
