import {NgModule} from '@angular/core';
import {MaxHeapChartDownloader} from './max_heap_chart_downloader';

/** @deprecated Import the standalone MaxHeapChartDownloader component directly instead. */
@NgModule({
  imports: [MaxHeapChartDownloader],
  exports: [MaxHeapChartDownloader],
})
export class MaxHeapChartDownloaderModule {}
