import {NgModule} from '@angular/core';

import {MemoryProfileSummary} from './memory_profile_summary';

@NgModule({
  imports: [MemoryProfileSummary],
  exports: [MemoryProfileSummary]
})
export class MemoryProfileSummaryModule {
}
