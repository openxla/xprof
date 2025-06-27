import {NgModule} from '@angular/core';

import {DownloadHlo} from './download_hlo';

@NgModule({
  imports: [
    DownloadHlo,
  ],
  exports: [DownloadHlo],
})
export class DownloadHloModule {
}
