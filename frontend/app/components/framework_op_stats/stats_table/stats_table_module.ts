import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';

import {StatsTable} from './stats_table';

@NgModule({
  imports: [
    CommonModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    ChartModule,
    StatsTable,
  ],
  exports: [StatsTable],
})
export class StatsTableModule {}
