import {NgModule} from '@angular/core';
import {InferenceProfile} from './inference_profile';

/** @deprecated Import the standalone InferenceProfile component directly instead. */
@NgModule({
  imports: [InferenceProfile],
  exports: [InferenceProfile],
})
export class InferenceProfileModule {}
