import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatLegacyOptionModule} from '@angular/material/core';
import {MatLegacyFormFieldModule} from '@angular/material/form-field';
import {MatLegacySelectModule} from '@angular/material/select';

import {CategoryFilter} from './category_filter';

/** A category filter module. */
@NgModule({
  declarations: [CategoryFilter],
  imports: [
    CommonModule,
    MatLegacyOptionModule,
    MatLegacyFormFieldModule,
    MatLegacySelectModule,
  ],
  exports: [CategoryFilter],
})
export class CategoryFilterModule {
}
