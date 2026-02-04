import {NgModule} from '@angular/core';

import {CaptureProfileDialog} from './capture_profile_dialog';

/** A capture profile dialog module. */
@NgModule({
  imports: [
    CaptureProfileDialog,
  ],
  exports: [CaptureProfileDialog]
})
export class CaptureProfileDialogModule {
}
