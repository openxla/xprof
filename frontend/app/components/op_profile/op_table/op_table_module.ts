import {NgModule} from '@angular/core';
import {OpTable} from './op_table';

/** An op table view module. */
@NgModule({
  imports: [OpTable],
  exports: [OpTable],
})
export class OpTableModule {
}
