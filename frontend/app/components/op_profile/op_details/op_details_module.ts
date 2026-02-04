import {NgModule} from '@angular/core';
import {OpDetails} from './op_details';

/** An op details view module. */
@NgModule({
  imports: [
    OpDetails,
  ],
  exports: [OpDetails]
})
export class OpDetailsModule {
}
