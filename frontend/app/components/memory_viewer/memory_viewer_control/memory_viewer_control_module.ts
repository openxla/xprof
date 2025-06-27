import {NgModule} from '@angular/core';

import {MemoryViewerControl} from './memory_viewer_control';

@NgModule({
  imports: [MemoryViewerControl],
  exports: [MemoryViewerControl],
})
export class MemoryViewerControlModule {
}
