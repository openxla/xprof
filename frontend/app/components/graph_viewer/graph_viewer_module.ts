import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatChipsModule} from '@angular/material/chips';
import {MatOptionModule} from '@angular/material/core';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSidenavModule} from '@angular/material/sidenav';
import {MatSnackBarModule} from '@angular/material/snack-bar';
import {MatTooltipModule} from '@angular/material/tooltip';
import {DownloadHloModule} from 'org_xprof/frontend/app/components/controls/download_hlo/download_hlo_module';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';
import {GraphConfigModule} from 'org_xprof/frontend/app/components/graph_viewer/graph_config/graph_config_module';
import {HloTextViewModule} from 'org_xprof/frontend/app/components/graph_viewer/hlo_text_view/hlo_text_view_module';
import {OpDetailsModule} from 'org_xprof/frontend/app/components/op_profile/op_details/op_details_module';
import {SourceMapperModule} from 'org_xprof/frontend/app/components/source_mapper/source_mapper_module';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';

import {GraphViewer} from './graph_viewer';

@NgModule({
  imports: [
    CommonModule,
    DiagnosticsViewModule,
    MatOptionModule,
    MatProgressBarModule,
    MatSidenavModule,
    PipesModule,
    GraphConfigModule,
    HloTextViewModule,
    OpDetailsModule,
    MatProgressSpinnerModule,
    MatSnackBarModule,
    DownloadHloModule,
    MatExpansionModule,
    SourceMapperModule,
    MatChipsModule,
    MatTooltipModule,
  ],
  declarations: [GraphViewer],
  exports: [GraphViewer]
})
export class GraphViewerModule {
}
