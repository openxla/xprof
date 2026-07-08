import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatButtonToggleModule} from '@angular/material/button-toggle';
import {MatCardModule} from '@angular/material/card';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSelectModule} from '@angular/material/select';
import {MatSortModule} from '@angular/material/sort';
import {MatTableModule} from '@angular/material/table';
import {MatTabsModule} from '@angular/material/tabs';
import {MatTooltipModule} from '@angular/material/tooltip';

import {MemoryFlameGraph} from './flame_graph/flame_graph';
import {MemoryAnalysis} from './memory_analysis';
import {MemoryTreemap} from './treemap/treemap';

import {SearchableDropdown} from 'org_xprof/frontend/app/components/controls/searchable_dropdown/searchable_dropdown';

/**
 * An Angular module that bundles together components and dependencies
 * for interactive peak heap memory breakdown analysis.
 */
@NgModule({
  declarations: [MemoryAnalysis, MemoryFlameGraph, MemoryTreemap],
  imports: [
    CommonModule,
    MatCardModule,
    MatFormFieldModule,
    MatProgressBarModule,
    MatProgressSpinnerModule,
    MatSelectModule,
    MatSortModule,
    MatTableModule,
    MatTabsModule,
    MatIconModule,
    MatButtonModule,
    MatButtonToggleModule,
    MatTooltipModule,
    MatInputModule,
    SearchableDropdown,
  ],
  exports: [MemoryAnalysis],
})
export class MemoryAnalysisModule {}
