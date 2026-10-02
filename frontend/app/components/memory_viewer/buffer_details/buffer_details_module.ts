import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatCardModule} from '@angular/material/card';

import {BufferDetails} from './buffer_details';

/** A buffer details view module. */
@NgModule({
  imports: [MatCardModule, MatButtonModule, BufferDetails],
  exports: [BufferDetails],
})
export class BufferDetailsModule {}
