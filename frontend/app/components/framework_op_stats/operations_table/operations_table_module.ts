import {NgModule} from '@angular/core';
import {OperationsTable} from './operations_table';

/** @deprecated Import the standalone OperationsTable component directly instead. */
@NgModule({
  imports: [OperationsTable],
  exports: [OperationsTable],
})
export class OperationsTableModule {}
