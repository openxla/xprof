import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatSidenavModule} from '@angular/material/sidenav';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';

import {MegascalePerfetto} from './megascale_perfetto';

@NgModule({
  declarations: [MegascalePerfetto],
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatSidenavModule,
    PipesModule,
  ],
  exports: [MegascalePerfetto],
})
export class MegascalePerfettoModule {}
