import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatExpansionModule} from '@angular/material/expansion';

import {Message} from './message';
import {StackFrameSnippetModule} from './stack_frame_snippet_module';
import {StackTraceSnippet} from './stack_trace_snippet';

/** A module to show code snippets for a stack trace. */
@NgModule({
  declarations: [StackTraceSnippet],
  exports: [StackTraceSnippet],
  imports: [CommonModule, MatExpansionModule, StackFrameSnippetModule, Message]
})
export class StackTraceSnippetModule {
}
