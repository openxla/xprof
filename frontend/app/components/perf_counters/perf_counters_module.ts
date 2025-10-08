import {NgModule} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {TableModule} from 'org_xprof/frontend/app/components/chart/table/table_module';
import {CategoryFilterModule} from 'org_xprof/frontend/app/components/controls/category_filter/category_filter_module';
import {ExportAsCsvModule} from 'org_xprof/frontend/app/components/controls/export_as_csv/export_as_csv_module';
import {StringFilterModule} from 'org_xprof/frontend/app/components/controls/string_filter/string_filter_module';

import {PerfCounters} from './perf_counters';

/** A perf counters module. */
@NgModule({
  declarations: [PerfCounters],
  imports: [
    StringFilterModule,
    CategoryFilterModule,
    ExportAsCsvModule,
    TableModule,
    MatCheckboxModule,
    FormsModule,
  ],
  exports: [PerfCounters],
})
export class PerfCountersModule {}
