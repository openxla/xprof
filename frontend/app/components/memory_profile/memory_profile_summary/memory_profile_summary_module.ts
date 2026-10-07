import {NgModule} from '@angular/core';
import {MemoryProfileSummary} from './memory_profile_summary';

/** @deprecated Import the standalone MemoryProfileSummary component directly instead. */
@NgModule({
  imports: [MemoryProfileSummary],
  exports: [MemoryProfileSummary],
})
export class MemoryProfileSummaryModule {}
