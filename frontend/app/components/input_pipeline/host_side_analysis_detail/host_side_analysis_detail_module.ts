import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatExpansionModule} from '@angular/material/expansion';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';

import {HostSideAnalysisDetail} from './host_side_analysis_detail';

@NgModule({
  imports: [
    CommonModule,
    MatExpansionModule,
    ChartModule,
    HostSideAnalysisDetail,
  ],
  exports: [HostSideAnalysisDetail],
})
export class HostSideAnalysisDetailModule {}
