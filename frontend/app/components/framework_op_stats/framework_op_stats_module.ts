import {NgModule} from '@angular/core';
import {FrameworkOpStats} from './framework_op_stats';

/** @deprecated Import the standalone FrameworkOpStats component directly instead. */
@NgModule({
  imports: [FrameworkOpStats],
  exports: [FrameworkOpStats],
})
export class FrameworkOpStatsModule {}
