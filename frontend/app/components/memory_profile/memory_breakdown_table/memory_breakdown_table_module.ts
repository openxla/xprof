import {NgModule} from '@angular/core';
import {MemoryBreakdownTable} from './memory_breakdown_table';

/** @deprecated Import the standalone MemoryBreakdownTable component directly instead. */
@NgModule({
  imports: [MemoryBreakdownTable],
  exports: [MemoryBreakdownTable],
})
export class MemoryBreakdownTableModule {}
