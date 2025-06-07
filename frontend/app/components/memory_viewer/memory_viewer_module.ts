import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatSidenavModule} from '@angular/material/sidenav';
import {BufferDetailsModule} from 'org_xprof/frontend/app/components/memory_viewer/buffer_details/buffer_details_module';
import {MemoryViewerControlModule} from 'org_xprof/frontend/app/components/memory_viewer/memory_viewer_control/memory_viewer_control_module';
import {MemoryViewerMainModule} from 'org_xprof/frontend/app/components/memory_viewer/memory_viewer_main/memory_viewer_main_module';

import {MemoryViewer} from './memory_viewer';

/** A memory viewer module. */
@NgModule({
  declarations: [MemoryViewer],
  imports: [
    MemoryViewerMainModule,
    MemoryViewerControlModule,
    BufferDetailsModule,
    CommonModule,
    MatProgressBarModule,
    MatSidenavModule,
  ],
  exports: [MemoryViewer]
})
export class MemoryViewerModule {
}
