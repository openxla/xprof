import {NgModule} from '@angular/core';

import {OrgChart} from './org_chart';

/** An organization chart view module. */
@NgModule({
  imports: [OrgChart],
  exports: [OrgChart],
})
export class OrgChartModule {}
