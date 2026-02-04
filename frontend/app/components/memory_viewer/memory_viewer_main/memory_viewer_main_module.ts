import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MemoryViewerMain} from './memory_viewer_main';

/** A memory viewer module. */
@NgModule({
  imports: [
    CommonModule,
    MemoryViewerMain,
  ],
  exports: [MemoryViewerMain]
})
export class MemoryViewerMainModule {
}
