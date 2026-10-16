import {NgModule} from '@angular/core';
import {StackTracePage} from './stack_trace_page';

/** @deprecated Import the standalone StackTracePage component directly instead. */
@NgModule({
  imports: [StackTracePage],
  exports: [StackTracePage],
})
export class StackTracePageModule {}
