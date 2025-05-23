import {NgModule} from '@angular/core';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';

import {MegascaleStats} from './megascale_stats';

/** A Megascale Stats module. */
@NgModule({
  declarations: [MegascaleStats],
  imports: [
    ChartModule,
  ],
  exports: [MegascaleStats],
})
export class MegascaleStatsModule {
}
