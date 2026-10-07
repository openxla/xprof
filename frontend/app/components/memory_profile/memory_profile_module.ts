import {NgModule} from '@angular/core';
import {MemoryProfile} from './memory_profile';

/** @deprecated Import the standalone MemoryProfile component directly instead. */
@NgModule({
  imports: [MemoryProfile],
  exports: [MemoryProfile],
})
export class MemoryProfileModule {}
