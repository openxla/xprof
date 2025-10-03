import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';

import {ViewArchitecture} from './view_architecture';

/**
 * A view-architecture button module.
 * This component exposes a button to generate a graphviz URL for the TPU
 * utilization viewer based on the used device architecture in the program code
 */
@NgModule({
  declarations: [ViewArchitecture],
  imports: [CommonModule, MatIconModule],
  exports: [ViewArchitecture],
})
export class ViewArchitectureModule {}
