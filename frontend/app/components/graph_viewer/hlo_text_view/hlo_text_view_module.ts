import {NgModule} from '@angular/core';

import {HloTextView} from './hlo_text_view';

/** NgModule shim for backwards compatibility with non-standalone callers. */
@NgModule({
  imports: [HloTextView],
  exports: [HloTextView],
})
export class HloTextViewModule {}
