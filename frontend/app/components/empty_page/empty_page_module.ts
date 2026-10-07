import {NgModule} from '@angular/core';
import {EmptyPage} from './empty_page';

/** @deprecated Import the standalone EmptyPage component directly instead. */
@NgModule({
  imports: [EmptyPage],
  exports: [EmptyPage],
})
export class EmptyPageModule {}
