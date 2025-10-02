import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {CategoryFilterModule} from 'org_xprof/frontend/app/components/controls/category_filter/category_filter_module';
import {ExportAsCsvModule} from 'org_xprof/frontend/app/components/controls/export_as_csv/export_as_csv_module';
import {ViewArchitectureModule} from 'org_xprof/frontend/app/components/controls/view_architecture/view_architecture_module';

import {UtilizationViewer} from './utilization_viewer';

/** Utilization viewer module. */
@NgModule({
  declarations: [UtilizationViewer],
  imports: [
    ChartModule,
    CategoryFilterModule,
    ExportAsCsvModule,
    CommonModule,
    ViewArchitectureModule,
  ],
  exports: [UtilizationViewer],
})
export class UtilizationViewerModule {
}
