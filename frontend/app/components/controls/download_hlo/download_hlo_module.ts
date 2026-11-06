import {CommonModule} from '@angular/common';
import {CUSTOM_ELEMENTS_SCHEMA, NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

import {DownloadHlo} from './download_hlo';

@NgModule({
  imports: [
    CommonModule,
    MatIconModule,
    MatTooltipModule,
  ],
  declarations: [DownloadHlo],
  exports: [DownloadHlo],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class DownloadHloModule {
}
