import {NgModule} from '@angular/core';
import {OpTableEntry} from './op_table_entry';

/** @deprecated Import the standalone OpTableEntry component directly instead. */
@NgModule({
  imports: [OpTableEntry],
  exports: [OpTableEntry],
})
export class OpTableEntryModule {}
