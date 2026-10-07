import {NgModule} from '@angular/core';
import {HostOp} from './host_op';

/** @deprecated Import the standalone HostOp component directly instead. */
@NgModule({
  imports: [HostOp],
  exports: [HostOp],
})
export class HostOpModule {}
