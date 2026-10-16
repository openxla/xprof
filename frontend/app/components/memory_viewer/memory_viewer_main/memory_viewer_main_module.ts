import {NgModule} from '@angular/core';
import {MemoryViewerMain} from './memory_viewer_main';

/** @deprecated Import the standalone MemoryViewerMain component directly instead. */
@NgModule({
  imports: [MemoryViewerMain],
  exports: [MemoryViewerMain],
})
export class MemoryViewerMainModule {}
