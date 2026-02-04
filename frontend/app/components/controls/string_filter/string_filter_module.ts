import {NgModule} from '@angular/core';

import {StringFilter} from './string_filter';

/** A string filter module. */
@NgModule({
  imports: [StringFilter],
  exports: [StringFilter],
})
export class StringFilterModule {
}
