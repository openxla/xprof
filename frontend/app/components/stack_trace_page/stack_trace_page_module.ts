import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {StackTraceSnippetModule} from 'org_xprof/frontend/app/components/stack_trace_snippet/stack_trace_snippet_module';

import {StackTracePage} from './stack_trace_page';

/** The stack trace page module. */
@NgModule({
  declarations: [StackTracePage],
  imports: [CommonModule, StackTraceSnippetModule],
  exports: [StackTracePage],
})
export class StackTracePageModule {
}
