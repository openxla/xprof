import {NgModule} from '@angular/core';

import {ExportAsCsv} from './export_as_csv';

/** A export-to-csv button module. */
@NgModule({
  imports: [ExportAsCsv],
  exports: [ExportAsCsv],
})
export class ExportAsCsvModule {
}
