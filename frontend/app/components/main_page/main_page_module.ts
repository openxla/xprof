import {NgModule} from '@angular/core';
import {RouterModule} from '@angular/router';
import {MainPage} from './main_page';
import {routes} from './routes';

export {routes};

/** A main page module. */
// TODO(xprof): Remove this module once all consumers import the standalone component directly.
@NgModule({
  imports: [MainPage, RouterModule.forRoot(routes)],
  exports: [MainPage],
})
export class MainPageModule {}
