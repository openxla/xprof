import {NgModule} from '@angular/core';
import {SafePipe} from './safe_pipe';

/** @deprecated Import the standalone SafePipe pipe directly instead. */
@NgModule({
  imports: [SafePipe],
  exports: [SafePipe],
})
export class PipesModule {}
