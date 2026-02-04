import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';

import {StepTimeGraph} from './step_time_graph';

@NgModule({
  imports: [MatCardModule, StepTimeGraph],
  exports: [StepTimeGraph]
})
export class StepTimeGraphModule {
}
