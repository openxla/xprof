import {NgModule} from '@angular/core';
import {MegascalePerfetto} from './megascale_perfetto';

/** @deprecated Import the standalone MegascalePerfetto component directly instead. */
@NgModule({
  imports: [MegascalePerfetto],
  exports: [MegascalePerfetto],
})
export class MegascalePerfettoModule {}
