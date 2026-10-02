import {NgModule} from '@angular/core';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';

import {DeviceSideAnalysisDetail} from './device_side_analysis_detail';

@NgModule({
  imports: [ChartModule, DeviceSideAnalysisDetail],
  exports: [DeviceSideAnalysisDetail],
})
export class DeviceSideAnalysisDetailModule {}
