import {NgModule} from '@angular/core';

import {CategoryFilter} from './category_filter';

/** A category filter module. */
@NgModule({
  imports: [CategoryFilter],
  exports: [CategoryFilter],
})
export class CategoryFilterModule {
}
