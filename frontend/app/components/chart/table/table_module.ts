import {NgModule} from '@angular/core';

import {Table} from './table';

/** A table view module. */
@NgModule({
  imports: [Table],
  exports: [Table],
})
export class TableModule {
}
