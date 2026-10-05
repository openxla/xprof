import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatOptionModule} from '@angular/material/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatSelectModule} from '@angular/material/select';

import {CategoryFilter} from './category_filter';

/** A category filter module. */
@NgModule({
  imports: [
    CommonModule,
    MatOptionModule,
    MatFormFieldModule,
    MatSelectModule,
    CategoryFilter,
  ],
  exports: [CategoryFilter],
})
export class CategoryFilterModule {}
