import {NgModule} from '@angular/core';

import {FrameworkOpStats} from './framework_op_stats';

/** An op profile module. */
@NgModule({
  imports: [FrameworkOpStats],
  exports: [FrameworkOpStats]
})
export class FrameworkOpStatsModule {
}
