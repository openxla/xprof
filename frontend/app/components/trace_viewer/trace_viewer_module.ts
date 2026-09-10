import {NgModule} from '@angular/core';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';

import {FilterChips} from './filter_chips';
import {FilterInput} from './filter_input';
import {TraceViewer} from './trace_viewer';

/** A trace viewer module for backwards compatibility with non-standalone callers. */
@NgModule({
  imports: [TraceViewer, FilterChips, FilterInput],
  providers: [DataServiceV2],
  exports: [TraceViewer, FilterChips, FilterInput],
})
export class TraceViewerModule {}
