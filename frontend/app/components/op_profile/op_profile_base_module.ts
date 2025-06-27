import {NgModule} from '@angular/core';

import {OpProfileBase} from './op_profile_base';

/** An op profile module. */
@NgModule({
  imports: [
    OpProfileBase,
  ],
  exports: [OpProfileBase]
})
export class OpProfileBaseModule {
}
