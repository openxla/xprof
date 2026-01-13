import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatCardModule} from '@angular/material/card';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatInputModule} from '@angular/material/input';

import {MemoryTimelineGraph} from './memory_timeline_graph';

@NgModule({
  declarations: [MemoryTimelineGraph],
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
  ],
  exports: [MemoryTimelineGraph]
})
export class MemoryTimelineGraphModule {
}
