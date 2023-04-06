import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatLegacyCardModule} from '@angular/material/card';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

import {PerformanceSummary} from './performance_summary';

@NgModule({
  declarations: [PerformanceSummary],
  imports: [
    CommonModule,
    MatLegacyCardModule,
    MatIconModule,
    MatTooltipModule,
  ],
  exports: [PerformanceSummary]
})
export class PerformanceSummaryModule {
}
