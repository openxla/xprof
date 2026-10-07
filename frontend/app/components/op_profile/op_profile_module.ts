import {NgModule} from '@angular/core';
import {OpProfile} from './op_profile';

/** @deprecated Import the standalone OpProfile component directly instead. */
@NgModule({
  imports: [OpProfile],
  exports: [OpProfile],
})
export class OpProfileModule {}
