import {NgModule} from '@angular/core';
import {LabsComponent} from './labs';

/** @deprecated Import the standalone LabsComponent component directly instead. */
@NgModule({
  imports: [LabsComponent],
  exports: [LabsComponent],
})
export class LabsModule {}
