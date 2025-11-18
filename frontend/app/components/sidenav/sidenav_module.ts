import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatOptionModule} from '@angular/material/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatSelectModule} from '@angular/material/select';
import {MatTooltipModule} from '@angular/material/tooltip';
import {CaptureProfileModule} from 'org_xprof/frontend/app/components/capture_profile/capture_profile_module';
import {BufferDetailsModule} from 'org_xprof/frontend/app/components/memory_viewer/buffer_details/buffer_details_module';
import {OpDetailsModule} from 'org_xprof/frontend/app/components/op_profile/op_details/op_details_module';
import {PodViewerDetailsModule} from 'org_xprof/frontend/app/components/pod_viewer/pod_viewer_details/pod_viewer_details_module';

import {SideNav} from './sidenav';

/** A side navigation module. */
@NgModule({
  declarations: [SideNav],
  imports: [
    CommonModule,
    MatButtonModule,
    FormsModule,
    MatCheckboxModule,
    MatFormFieldModule,
    MatIconModule,
    MatInputModule,
    MatSelectModule,
    MatOptionModule,
    MatTooltipModule,
    BufferDetailsModule,
    CaptureProfileModule,
    OpDetailsModule,
    PodViewerDetailsModule,
  ],
  exports: [SideNav]
})
export class SideNavModule {
}
