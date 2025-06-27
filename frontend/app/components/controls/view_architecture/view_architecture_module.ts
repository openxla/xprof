import {NgModule} from '@angular/core';

import {ViewArchitecture} from './view_architecture';

/**
 * A view-architecture button module.
 * This component exposes a button to generate a graphviz URL for the TPU
 * utilization viewer based on the used device architecture in the program code
 */
@NgModule({
  imports: [ViewArchitecture],
  exports: [ViewArchitecture],
})
export class ViewArchitectureModule {}
