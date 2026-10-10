import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';

import {ExportAsCsv} from './export_as_csv';

/** A export-to-csv button module. */
// TODO(xprof): Remove this module once all consumers have migrated to importing
// the standalone component directly.
@NgModule({
  imports: [CommonModule, MatIconModule, ExportAsCsv],
  exports: [ExportAsCsv],
})
export class ExportAsCsvModule {}
