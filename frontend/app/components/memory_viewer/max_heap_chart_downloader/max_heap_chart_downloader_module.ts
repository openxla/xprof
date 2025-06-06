import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';

import {MaxHeapChartDownloader} from './max_heap_chart_downloader';

@NgModule({
  imports: [CommonModule, MatButtonModule],
  declarations: [MaxHeapChartDownloader],
  exports: [MaxHeapChartDownloader],
})
export class MaxHeapChartDownloaderModule {}
