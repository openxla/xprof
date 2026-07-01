/**
 * @fileoverview Angular module for Memory Breakdown Analysis.
 */

import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatCardModule} from '@angular/material/card';
import {MatChipsModule} from '@angular/material/chips';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatSortModule} from '@angular/material/sort';
import {MatTableModule} from '@angular/material/table';

import {MemoryAnalysis} from './memory_analysis';

/** A memory analysis module. */
@NgModule({
  declarations: [MemoryAnalysis],
  imports: [
    CommonModule,
    MatCardModule,
    MatChipsModule,
    MatProgressBarModule,
    MatSortModule,
    MatTableModule,
  ],
  exports: [MemoryAnalysis],
})
export class MemoryAnalysisModule {}
