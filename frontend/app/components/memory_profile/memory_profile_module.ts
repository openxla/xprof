import {NgModule} from '@angular/core';
import {MemoryProfile} from './memory_profile';

@NgModule({
  imports: [MemoryProfile],
  exports: [MemoryProfile],
})
export class MemoryProfileModule {}
