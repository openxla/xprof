import {NgModule} from '@angular/core';
import {DownloadHlo} from './download_hlo';

/** @deprecated Import the standalone DownloadHlo component directly instead. */
@NgModule({
  imports: [DownloadHlo],
  exports: [DownloadHlo],
})
export class DownloadHloModule {}
