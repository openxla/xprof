import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatOptionModule} from '@angular/material/core';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatSelectModule} from '@angular/material/select';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {CategoryFilterModule} from 'org_xprof/frontend/app/components/controls/category_filter/category_filter_module';
import {ExportAsCsvModule} from 'org_xprof/frontend/app/components/controls/export_as_csv/export_as_csv_module';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';

import {MegascaleStats} from './megascale_stats';

/** A Megascale Stats module. */
@NgModule({
  declarations: [MegascaleStats],
  imports: [
    CommonModule,
    ChartModule,
    CategoryFilterModule,
    DiagnosticsViewModule,
    ExportAsCsvModule,
    FormsModule,
    MatExpansionModule,
    MatFormFieldModule,
    MatSelectModule,
    MatOptionModule,
  ],
  exports: [MegascaleStats],
})
export class MegascaleStatsModule {
}
