import {NgModule} from '@angular/core';
import {ExportAsCsv} from './export_as_csv';

@NgModule({
  imports: [ExportAsCsv],
  exports: [ExportAsCsv],
})
export class ExportAsCsvModule {}
