import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';

import {OpTableEntry} from './op_table_entry';

/** An op table entry view module. */
@NgModule({
  declarations: [OpTableEntry],
  imports: [CommonModule, MatIconModule],
  exports: [OpTableEntry]
})
export class OpTableEntryModule {
}
