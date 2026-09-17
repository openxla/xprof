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
  computed,
  contentChildren,
  forwardRef,
  inject,
  input,
  model,
  output,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {
  NAV_DRAWER_CONTEXT,
  NAV_GROUP_CONTEXT,
  RackNavigationItem,
} from './navigation_item';

/**
 * Navigation group component.
 * A collapsible section that coordinates child navigation items.
 * Pinned children remain visible even when collapsed.
 */
@Component({
  selector: 'rack-navigation-group',
  standalone: true,
  imports: [MatIconModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrls: ['./navigation_group.scss'],
  providers: [
    {
      provide: NAV_GROUP_CONTEXT,
      useExisting: forwardRef(() => RackNavigationGroup),
    },
  ],
  host: {
    '[class.empty]': 'isEmpty()',
    '[attr.empty]': 'isEmpty() ? "" : null',
    '[class.drawer-collapsed]': 'isDrawerCollapsed()',
    '[attr.drawer-collapsed]': 'isDrawerCollapsed() ? "" : null',
  },
  template: `
    <button
      type="button"
      class="group-header"
      tabindex="-1"
      (click)="toggleExpand()"
      [attr.aria-expanded]="expanded()">
      <span class="group-title">{{ label() }}</span>
      <mat-icon
        fontSet="material-symbols-outlined"
        class="expand-icon material-symbols-outlined"
        aria-hidden="true"
        [style.transform]="expanded() ? 'none' : 'rotate(-90deg)'">
        expand_more
      </mat-icon>
    </button>
    <div class="group-divider"></div>
    <div
      class="group-content"
      [class.expanded]="expanded()"
      [class.collapsed]="!expanded()">
      <ng-content></ng-content>
    </div>
  `,
})
export class RackNavigationGroup {
  readonly label = input<string>('');
  readonly expanded = model<boolean>(true);
  readonly drawerCollapsed = input<boolean, unknown>(false, {
    transform: booleanAttribute,
  });

  readonly expandedChangeKebab = output<boolean>({alias: 'expanded-change'});

  readonly items = contentChildren(RackNavigationItem);

  private readonly parentDrawer = inject(NAV_DRAWER_CONTEXT, {optional: true});

  readonly isDrawerCollapsed = computed(
    () => this.drawerCollapsed() || (this.parentDrawer?.collapsed() ?? false),
  );

  readonly isEmpty = computed(() => {
    if (!this.isDrawerCollapsed()) return false;
    if (this.expanded()) return false;
    const hasPinned = this.items().some((item) => item.pinned());
    return !hasPinned;
  });

  toggleExpand() {
    const next = !this.expanded();
    this.expanded.set(next);
    this.expandedChangeKebab.emit(next);
  }
}
