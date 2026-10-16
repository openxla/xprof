import {NgModule} from '@angular/core';
import {Table} from './table';

/** @deprecated Import the standalone Table component directly instead. */
@NgModule({
  imports: [Table],
  exports: [Table],
})
export class TableModule {}
