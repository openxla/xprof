import {NgModule} from '@angular/core';

import {StackFrameSnippet} from './stack_frame_snippet';

/** A module to show code snippets for a stack frame. */
@NgModule({
  imports: [StackFrameSnippet],
  exports: [StackFrameSnippet],
})
export class StackFrameSnippetModule {
}
