import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatTooltipModule} from '@angular/material/tooltip';

import {HloTextView} from './hlo_text_view';

@NgModule({
  imports: [
    CommonModule,
    MatExpansionModule,
    MatTooltipModule,
    MatButtonModule,
    MatProgressBarModule,
    MatFormFieldModule,
  ],
  declarations: [HloTextView],
  exports: [HloTextView],
})
export class HloTextViewModule {}
