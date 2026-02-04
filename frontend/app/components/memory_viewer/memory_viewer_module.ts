import {NgModule} from '@angular/core';

import {MemoryViewer} from './memory_viewer';

/** A memory viewer module. */
@NgModule({
  imports: [MemoryViewer],
  exports: [MemoryViewer]
})
export class MemoryViewerModule {
}
