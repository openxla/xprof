import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatAutocompleteModule} from '@angular/material/autocomplete';
import {MatButtonModule} from '@angular/material/button';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatChipsModule} from '@angular/material/chips';
import {MatDialogModule} from '@angular/material/dialog';
import {MatDividerModule} from '@angular/material/divider';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatTooltipModule} from '@angular/material/tooltip';
import {TraceViewerContainer} from 'org_xprof/frontend/app/components/trace_viewer_container/trace_viewer_container';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';

import {FilterChips} from 'org_xprof/frontend/app/components/trace_viewer/filter_chips/filter_chips';
import {FilterInput} from 'org_xprof/frontend/app/components/trace_viewer/filter_input/filter_input';
import {TraceViewer} from './trace_viewer';

/** A trace viewer module. */
@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    MatAutocompleteModule,
    MatButtonModule,
    MatCheckboxModule,
    MatChipsModule,
    MatDialogModule,
    MatDividerModule,
    MatIconModule,
    MatMenuModule,
    MatProgressBarModule,
    MatTooltipModule,
    PipesModule,
    TraceViewerContainer,
    TraceViewer,
    FilterChips,
    FilterInput,
  ],
  providers: [DataServiceV2],
  exports: [TraceViewer, FilterChips, FilterInput],
})
export class TraceViewerModule {}
