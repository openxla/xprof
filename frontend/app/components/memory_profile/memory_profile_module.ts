import {NgModule} from '@angular/core';
import {MemoryProfile} from './memory_profile';

/** A memory profile module. */
@NgModule({
  imports: [
    MemoryProfile,
  ],
  exports: [MemoryProfile],
})
export class MemoryProfileModule {
}
