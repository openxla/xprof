import {NgModule} from '@angular/core';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';

import {OperationsTable} from './operations_table';

@NgModule({
  imports: [ChartModule, OperationsTable],
  exports: [OperationsTable],
})
export class OperationsTableModule {}
