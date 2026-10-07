import {NgModule} from '@angular/core';
import {RunEnvironmentView} from './run_environment_view';

/** @deprecated Import the standalone RunEnvironmentView component directly instead. */
@NgModule({
  imports: [RunEnvironmentView],
  exports: [RunEnvironmentView],
})
export class RunEnvironmentViewModule {}
