import {NgModule} from '@angular/core';
import {OverviewPage} from './overview_page';

export {OverviewPage} from './overview_page';

@NgModule({
  imports: [OverviewPage],
  exports: [OverviewPage],
})
export class OverviewPageModule {}
