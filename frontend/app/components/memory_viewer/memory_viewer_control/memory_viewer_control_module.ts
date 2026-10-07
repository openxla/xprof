import {NgModule} from '@angular/core';
import {MemoryViewerControl} from './memory_viewer_control';

/** @deprecated Import the standalone MemoryViewerControl component directly instead. */
@NgModule({
  imports: [MemoryViewerControl],
  exports: [MemoryViewerControl],
})
export class MemoryViewerControlModule {}
