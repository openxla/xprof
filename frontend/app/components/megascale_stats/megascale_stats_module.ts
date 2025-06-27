import {NgModule} from '@angular/core';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {CategoryFilterModule} from 'org_xprof/frontend/app/components/controls/category_filter/category_filter_module';
import {ExportAsCsvModule} from 'org_xprof/frontend/app/components/controls/export_as_csv/export_as_csv_module';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';

import {MegascaleStats} from './megascale_stats';

/** A Megascale Stats module. */
@NgModule({
  declarations: [MegascaleStats],
  imports: [
    ChartModule,
    CategoryFilterModule,
    DiagnosticsViewModule,
    ExportAsCsvModule,
  ],
  exports: [MegascaleStats],
})
export class MegascaleStatsModule {
}
