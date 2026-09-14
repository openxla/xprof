import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';

import {MatButtonModule} from '@angular/material/button';
import {MatTooltipModule} from '@angular/material/tooltip';

import {StringFilter} from './string_filter';

/** A string filter module. */
@NgModule({
  imports: [
    CommonModule,
    MatButtonModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatTooltipModule,
    StringFilter,
  ],
  exports: [StringFilter],
})
export class StringFilterModule {}
