import {NgModule} from '@angular/core';
import {MemoryViewer} from './memory_viewer';

@NgModule({
  imports: [MemoryViewer],
  exports: [MemoryViewer],
})
export class MemoryViewerModule {}
