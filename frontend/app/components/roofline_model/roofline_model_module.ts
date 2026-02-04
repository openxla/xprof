import {NgModule} from '@angular/core';

import {RooflineModel} from './roofline_model';

/** A roofline model module. */
@NgModule({
  imports: [RooflineModel],
  exports: [RooflineModel],
})
export class RooflineModelModule {
}
