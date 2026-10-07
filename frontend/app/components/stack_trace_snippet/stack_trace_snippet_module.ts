import {NgModule} from '@angular/core';
import {StackTraceSnippet} from './stack_trace_snippet';

/**
 * @deprecated Import the standalone `StackTraceSnippet` component directly
 *     instead.
 */
@NgModule({
  imports: [StackTraceSnippet],
  exports: [StackTraceSnippet],
})
export class StackTraceSnippetModule {}
