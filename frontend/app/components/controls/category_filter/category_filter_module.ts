import {NgModule} from '@angular/core';
import {CategoryFilter} from './category_filter';

/** @deprecated Import the standalone CategoryFilter component directly instead. */
@NgModule({
  imports: [CategoryFilter],
  exports: [CategoryFilter],
})
export class CategoryFilterModule {}
