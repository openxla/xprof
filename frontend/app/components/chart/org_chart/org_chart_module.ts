import {NgModule} from '@angular/core';
import {OrgChart} from './org_chart';

/** @deprecated Import the standalone OrgChart component directly instead. */
@NgModule({
  imports: [OrgChart],
  exports: [OrgChart],
})
export class OrgChartModule {}
