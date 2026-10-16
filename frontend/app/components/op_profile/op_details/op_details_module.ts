import {NgModule} from '@angular/core';
import {OpDetails} from './op_details';

/** @deprecated Import the standalone OpDetails component directly instead. */
@NgModule({
  imports: [OpDetails],
  exports: [OpDetails],
})
export class OpDetailsModule {}
