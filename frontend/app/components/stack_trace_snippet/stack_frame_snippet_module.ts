import {NgModule} from '@angular/core';
import {StackFrameSnippet} from './stack_frame_snippet';

/** @deprecated Import standalone StackFrameSnippet directly. */
@NgModule({
  imports: [StackFrameSnippet],
  exports: [StackFrameSnippet],
})
export class StackFrameSnippetModule {}
