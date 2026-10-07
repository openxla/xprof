import {NgModule} from '@angular/core';
import {MemoryViewer} from './memory_viewer';

/** @deprecated Import the standalone MemoryViewer component directly instead. */
@NgModule({
  imports: [MemoryViewer],
  exports: [MemoryViewer],
})
export class MemoryViewerModule {}
