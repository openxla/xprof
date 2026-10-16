import {NgModule} from '@angular/core';
import {MegascaleStats} from './megascale_stats';

/** @deprecated Import the standalone MegascaleStats component directly instead. */
@NgModule({
  imports: [MegascaleStats],
  exports: [MegascaleStats],
})
export class MegascaleStatsModule {}
