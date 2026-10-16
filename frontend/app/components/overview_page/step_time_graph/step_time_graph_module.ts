import {NgModule} from '@angular/core';
import {StepTimeGraph} from './step_time_graph';

/** @deprecated Import the standalone StepTimeGraph component directly instead. */
@NgModule({
  imports: [StepTimeGraph],
  exports: [StepTimeGraph],
})
export class StepTimeGraphModule {}
