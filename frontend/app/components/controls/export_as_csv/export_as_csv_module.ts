import {NgModule} from '@angular/core';
import {ExportAsCsv} from './export_as_csv';

/** @deprecated Import the standalone ExportAsCsv component directly instead. */
@NgModule({
  imports: [ExportAsCsv],
  exports: [ExportAsCsv],
})
export class ExportAsCsvModule {}
