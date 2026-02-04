import {NgModule} from '@angular/core';

import {CaptureProfile} from './capture_profile';

/** A capture profile view module. */
@NgModule({
  imports: [
    CaptureProfile,
  ],
  exports: [CaptureProfile]
})
export class CaptureProfileModule {
}
