import {NgModule} from '@angular/core';
import {InputPipeline} from './input_pipeline';

/** @deprecated Import the standalone InputPipeline component directly instead. */
@NgModule({
  imports: [InputPipeline],
  exports: [InputPipeline],
})
export class InputPipelineModule {}
