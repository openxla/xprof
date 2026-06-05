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

import {FilterChips} from './filter_chips';
import {FilterInput} from './filter_input';
import {TraceViewer} from './trace_viewer';

/** A trace viewer module. */
@NgModule({
  declarations: [TraceViewer, FilterChips, FilterInput],
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
  ],
  providers: [DataServiceV2],
  exports: [TraceViewer],
})
export class TraceViewerModule {}
