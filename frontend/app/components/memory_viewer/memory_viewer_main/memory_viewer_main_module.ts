import {AngularSplitModule} from 'angular-split';
import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatDividerModule} from '@angular/material/divider';
import {MatIconModule} from '@angular/material/icon';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatTooltipModule} from '@angular/material/tooltip';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';
import {MaxHeapChartModule} from 'org_xprof/frontend/app/components/memory_viewer/max_heap_chart/max_heap_chart_module';
import {ProgramOrderChartModule} from 'org_xprof/frontend/app/components/memory_viewer/program_order_chart/program_order_chart_module';
import {StackTraceSnippetModule} from 'org_xprof/frontend/app/components/stack_trace_snippet/stack_trace_snippet_module';

import {MemoryViewerMain} from './memory_viewer_main';

/** A memory viewer module. */
@NgModule({
  declarations: [MemoryViewerMain],
  imports: [
    AngularSplitModule,
    CommonModule,
    DiagnosticsViewModule,
    FormsModule,
    MatDividerModule,
    MaxHeapChartModule,
    MatCheckboxModule,
    MatIconModule,
    MatSlideToggleModule,
    MatTooltipModule,
    ProgramOrderChartModule,
    StackTraceSnippetModule,
  ],
  exports: [MemoryViewerMain]
})
export class MemoryViewerMainModule {
}
