import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatSelectModule} from '@angular/material/select';
import {MatSidenavModule} from '@angular/material/sidenav';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatTooltipModule} from '@angular/material/tooltip';
import {AngularSplitModule} from 'angular-split';
import {SourceMapperModule} from 'org_xprof/frontend/app/components/source_mapper/source_mapper_module';

import {OpProfileBase} from './op_profile_base';
import {OpTableModule} from './op_table/op_table_module';

/** An op profile module. */
@NgModule({
  declarations: [OpProfileBase],
  imports: [
    AngularSplitModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatSlideToggleModule,
    OpTableModule,
    MatIconModule,
    MatTooltipModule,
    MatSidenavModule,
    CommonModule,
    SourceMapperModule,
  ],
  exports: [OpProfileBase]
})
export class OpProfileBaseModule {
}
