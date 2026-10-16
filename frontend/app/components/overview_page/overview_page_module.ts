import {NgModule} from '@angular/core';
import {OverviewPage} from './overview_page';

export {OverviewPage} from './overview_page';

/** @deprecated Import the standalone OverviewPage component directly instead. */
@NgModule({
  imports: [OverviewPage],
  exports: [OverviewPage],
})
export class OverviewPageModule {}
