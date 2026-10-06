import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTableModule} from '@angular/material/table';
import {MatTooltipModule} from '@angular/material/tooltip';

import {OpTable} from './op_table';

/** An op table view module. */
@NgModule({
  declarations: [OpTable],
  imports: [CommonModule, MatIconModule, MatTableModule, MatTooltipModule],
  exports: [OpTable],
})
export class OpTableModule {}
