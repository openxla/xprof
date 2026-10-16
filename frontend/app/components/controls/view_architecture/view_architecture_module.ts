import {NgModule} from '@angular/core';
import {ViewArchitecture} from './view_architecture';

/** @deprecated Import the standalone ViewArchitecture component directly instead. */
@NgModule({
  imports: [ViewArchitecture],
  exports: [ViewArchitecture],
})
export class ViewArchitectureModule {}
