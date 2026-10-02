import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatExpansionModule} from '@angular/material/expansion';

import {NormalizedAcceleratorPerformanceView} from './normalized_accelerator_performance_view';

/** A normalized accelerator performance view module. */
@NgModule({
  imports: [
    CommonModule,
    MatExpansionModule,
    NormalizedAcceleratorPerformanceView,
  ],
  exports: [NormalizedAcceleratorPerformanceView],
})
export class NormalizedAcceleratorPerformanceViewModule {}
