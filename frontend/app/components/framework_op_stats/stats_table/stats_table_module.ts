import {NgModule} from '@angular/core';
import {StatsTable} from './stats_table';

/** @deprecated Import the standalone StatsTable component directly instead. */
@NgModule({
  imports: [StatsTable],
  exports: [StatsTable],
})
export class StatsTableModule {}
