import {NgModule} from '@angular/core';
import {StringFilter} from './string_filter';

/** @deprecated Import the standalone StringFilter component directly instead. */
@NgModule({
  imports: [StringFilter],
  exports: [StringFilter],
})
export class StringFilterModule {}
