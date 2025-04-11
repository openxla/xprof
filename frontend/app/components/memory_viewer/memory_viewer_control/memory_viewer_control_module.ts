import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatOptionModule} from '@angular/material/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatSelectModule} from '@angular/material/select';

import {MemoryViewerControl} from './memory_viewer_control';

@NgModule({
  imports: [
    CommonModule,
    MatFormFieldModule, MatSelectModule, MatOptionModule, FormsModule
  ],
  declarations: [MemoryViewerControl],
  exports: [MemoryViewerControl],
})
export class MemoryViewerControlModule {
}
