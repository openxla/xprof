import {NgModule} from '@angular/core';
import {SafePipe} from './safe_pipe';

@NgModule({
  imports: [SafePipe],
  exports: [SafePipe],
})
export class PipesModule {}
