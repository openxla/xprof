/**
 * @license
 * Copyright 2026 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  booleanAttribute,
  ChangeDetectionStrategy,
  Component,
  forwardRef,
  input,
  model,
} from '@angular/core';
import {NAV_DRAWER_CONTEXT} from './navigation_item';

/**
 * Navigation drawer component.
 * The main container managing width, collapsed state, and classic modes.
 */
@Component({
  selector: 'rack-navigation-drawer',
  standalone: true,
  imports: [],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrls: ['./navigation_drawer.scss'],
  providers: [
    {
      provide: NAV_DRAWER_CONTEXT,
      useExisting: forwardRef(() => RackNavigationDrawer),
    },
  ],
  host: {
    '[class.collapsed]': 'collapsed()',
    '[attr.collapsed]': 'collapsed() ? "" : null',
    '[class.classic]': 'classic()',
    '[attr.classic]': 'classic() ? "" : null',
  },
  template: `
    <div class="drawer-content">
      <ng-content></ng-content>
    </div>
    <div class="drawer-footer"><ng-content select="[slot=footer],[footer]"></ng-content></div>
  `,
})
export class RackNavigationDrawer {
  readonly collapsed = model<boolean>(false);
  readonly classic = input<boolean, unknown>(false, {
    transform: booleanAttribute,
  });
}
