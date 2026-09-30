import {CommonModule} from '@angular/common';
import {CUSTOM_ELEMENTS_SCHEMA, NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSnackBarModule} from '@angular/material/snack-bar';

import {CaptureProfile} from './capture_profile';
import {CaptureProfileDialogModule} from './capture_profile_dialog/capture_profile_dialog_module';

/** A capture profile view module. */
@NgModule({
  declarations: [CaptureProfile],
  imports: [
    CommonModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    CaptureProfileDialogModule,
    MatSnackBarModule,
  ],
  exports: [CaptureProfile],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class CaptureProfileModule {}
