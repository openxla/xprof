import {NgModule} from '@angular/core';

import {HloTextView} from './hlo_text_view';

@NgModule({
  imports: [
    HloTextView,
  ],
  exports: [HloTextView],
})
export class HloTextViewModule {}
