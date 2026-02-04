import {NgModule} from '@angular/core';

import {StackTraceSnippet} from './stack_trace_snippet';

/** A module to show code snippets for a stack trace. */
@NgModule({
  imports: [StackTraceSnippet],
  exports: [StackTraceSnippet],
})
export class StackTraceSnippetModule {
}
