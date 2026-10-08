import {NgModule} from '@angular/core';

import {SideNav} from './sidenav';

/** A side navigation module. */
@NgModule({
  imports: [SideNav],
  exports: [SideNav],
})
export class SideNavModule {}
