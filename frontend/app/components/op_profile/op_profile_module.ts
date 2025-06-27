import {NgModule} from '@angular/core';

import {OpProfile} from './op_profile';

/** An op profile module. */
@NgModule({
  imports: [
    OpProfile,
  ],
  exports: [OpProfile],
})
export class OpProfileModule {
}
