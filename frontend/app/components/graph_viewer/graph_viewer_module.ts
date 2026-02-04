import {NgModule} from '@angular/core';

import {GraphViewer} from './graph_viewer';



@NgModule({
  imports: [
    GraphViewer,
  ],
  exports: [GraphViewer]
})
export class GraphViewerModule {
}
