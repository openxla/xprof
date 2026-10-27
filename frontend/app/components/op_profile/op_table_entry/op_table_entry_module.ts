import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatTooltipModule} from '@angular/material/tooltip';

import {OpTableEntry} from './op_table_entry';

/** An op table entry view module. */
@NgModule({
  declarations: [OpTableEntry],
  imports: [CommonModule, MatTooltipModule],
  exports: [OpTableEntry],
})
export class OpTableEntryModule {}
