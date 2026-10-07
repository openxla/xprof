import {NgModule} from '@angular/core';
import {HloTextView} from './hlo_text_view';

/** @deprecated Import the standalone HloTextView component directly instead. */
@NgModule({
  imports: [HloTextView],
  exports: [HloTextView],
})
export class HloTextViewModule {}
