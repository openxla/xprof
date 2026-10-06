import {CommonModule} from '@angular/common';
import {CUSTOM_ELEMENTS_SCHEMA, NgModule} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatDialogModule} from '@angular/material/dialog';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatInputModule} from '@angular/material/input';
import {MatSelectModule} from '@angular/material/select';
import {MatTooltipModule} from '@angular/material/tooltip';

import {CaptureProfileDialog} from './capture_profile_dialog';

/** A capture profile dialog module. */
@NgModule({
  declarations: [CaptureProfileDialog],
  imports: [
    CommonModule,
    FormsModule,
    MatDialogModule,
    MatExpansionModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatTooltipModule,
  ],
  exports: [CaptureProfileDialog],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class CaptureProfileDialogModule {}
