import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatOptionModule} from '@angular/material/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatSelectModule} from '@angular/material/select';
import {DownloadHloModule} from 'org_xprof/frontend/app/components/controls/download_hlo/download_hlo_module';
import {SearchableDropdown} from 'org_xprof/frontend/app/components/controls/searchable_dropdown/searchable_dropdown';

import {MemoryViewerControl} from './memory_viewer_control';

@NgModule({
  imports: [
    CommonModule,
    SearchableDropdown,
    DownloadHloModule,
    MatFormFieldModule,
    MatSelectModule,
    MatOptionModule,
  ],
  declarations: [MemoryViewerControl],
  exports: [MemoryViewerControl],
})
export class MemoryViewerControlModule {
}
