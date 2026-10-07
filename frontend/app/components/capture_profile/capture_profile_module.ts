import {NgModule} from '@angular/core';
import {CaptureProfile} from './capture_profile';

/** @deprecated Import the standalone CaptureProfile component directly instead. */
@NgModule({
  imports: [CaptureProfile],
  exports: [CaptureProfile],
})
export class CaptureProfileModule {}
