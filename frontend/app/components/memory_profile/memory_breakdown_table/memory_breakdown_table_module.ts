import {NgModule} from '@angular/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';

import {MemoryBreakdownTable} from './memory_breakdown_table';

@NgModule({
  imports: [
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MemoryBreakdownTable,
  ],
  exports: [MemoryBreakdownTable],
})
export class MemoryBreakdownTableModule {}
