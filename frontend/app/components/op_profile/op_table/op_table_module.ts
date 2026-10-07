import {NgModule} from '@angular/core';
import {OpTable} from './op_table';

/** @deprecated Import the standalone OpTable component directly instead. */
@NgModule({
  imports: [OpTable],
  exports: [OpTable],
})
export class OpTableModule {}
