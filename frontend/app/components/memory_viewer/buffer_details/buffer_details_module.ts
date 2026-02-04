import {NgModule} from '@angular/core';

import {BufferDetails} from './buffer_details';

/** A buffer details view module. */
@NgModule({
  imports: [
    BufferDetails,
  ],
  exports: [BufferDetails]
})
export class BufferDetailsModule {
}
