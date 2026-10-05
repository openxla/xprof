import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatSidenavModule} from '@angular/material/sidenav';
import {MatSlideToggleModule} from '@angular/material/slide-toggle';
import {MatTooltipModule} from '@angular/material/tooltip';
import {OpDetailsModule} from 'org_xprof/frontend/app/components/op_profile/op_details/op_details_module';
import {OpProfileBaseModule} from 'org_xprof/frontend/app/components/op_profile/op_profile_base_module';
import {OpTableModule} from 'org_xprof/frontend/app/components/op_profile/op_table/op_table_module';

import {OpProfile} from './op_profile';

/** An op profile module. */
@NgModule({
  imports: [
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatSidenavModule,
    MatSlideToggleModule,
    MatTooltipModule,
    OpTableModule,
    OpDetailsModule,
    CommonModule,
    OpProfileBaseModule,
    OpProfile,
  ],
  exports: [OpProfile],
})
export class OpProfileModule {}
