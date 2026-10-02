import {NgModule} from '@angular/core';
import {MatTooltipModule} from '@angular/material/tooltip';
import {OpTableEntryModule} from 'org_xprof/frontend/app/components/op_profile/op_table_entry/op_table_entry_module';

import {OpTable} from './op_table';

/** An op table view module. */
@NgModule({
  imports: [OpTableEntryModule, MatTooltipModule, OpTable],
  exports: [OpTable],
})
export class OpTableModule {}
