import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';

import {SideNav} from './sidenav';

/** A side navigation module. */
@NgModule({
  imports: [
    CommonModule,
    SideNav,
  ],
  exports: [SideNav]
})
export class SideNavModule {
}
