import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatOptionModule} from '@angular/material/core';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatSelectModule} from '@angular/material/select';
import {MatSidenavModule} from '@angular/material/sidenav';
import {TableModule} from 'org_xprof/frontend/app/components/chart/table/table_module';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';

import {InferenceProfile} from './inference_profile';

@NgModule({
  imports: [
    CommonModule,
    TableModule,
    MatOptionModule,
    MatSelectModule,
    MatSidenavModule,
    MatProgressBarModule,
    DiagnosticsViewModule,
  ],
  declarations: [InferenceProfile],
  exports: [InferenceProfile],
})
export class InferenceProfileModule {}
