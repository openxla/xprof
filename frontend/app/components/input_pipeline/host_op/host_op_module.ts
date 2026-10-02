import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatOptionModule} from '@angular/material/core';
import {MatDividerModule} from '@angular/material/divider';
import {MatSelectModule} from '@angular/material/select';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';

import {HostOp} from './host_op';

@NgModule({
  imports: [
    CommonModule,
    MatDividerModule,
    MatSelectModule,
    MatOptionModule,
    ChartModule,
    HostOp,
  ],
  exports: [HostOp],
})
export class HostOpModule {}
