import {NgModule} from '@angular/core';
import {OpProfileBase} from './op_profile_base';

/** @deprecated Import the standalone OpProfileBase component directly instead. */
@NgModule({
  imports: [OpProfileBase],
  exports: [OpProfileBase],
})
export class OpProfileBaseModule {}
