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
  AfterViewInit,
  booleanAttribute,
  ChangeDetectionStrategy,
  Component,
  computed,
  ElementRef,
  inject,
  InjectionToken,
  input,
  model,
  output,
  signal,
  Signal,
  viewChild,
} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

/**
 * Interface representing the parent navigation group context.
 */
export interface NavigationGroupContext {
  readonly expanded: Signal<boolean>;
}

/**
 * Interface representing the parent navigation drawer context.
 */
export interface NavigationDrawerContext {
  readonly collapsed: Signal<boolean>;
}

/**
 * Injection token for child navigation items to query parent group state.
 */
export const NAV_GROUP_CONTEXT = new InjectionToken<NavigationGroupContext>(
  'NAV_GROUP_CONTEXT',
);

/**
 * Injection token for child navigation items to query parent drawer state.
 */
export const NAV_DRAWER_CONTEXT = new InjectionToken<NavigationDrawerContext>(
  'NAV_DRAWER_CONTEXT',
);

/**
 * Navigation item component.
 * Individual navigation link with icon, label, active state, and a pin toggle.
 */
@Component({
  selector: 'rack-navigation-item',
  standalone: true,
  imports: [MatIconModule, MatTooltipModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrls: ['./navigation_item.scss'],
  host: {
    '[class.active]': 'active()',
    '[attr.active]': 'active() ? "" : null',
    '[class.disabled]': 'disabled()',
    '[attr.disabled]': 'disabled() ? "" : null',
    '[class.pinned]': 'pinned()',
    '[class.collapsed-hidden]': 'isHiddenByGroup()',
    '[class.drawer-collapsed]': 'isDrawerCollapsed()',
    '[attr.aria-hidden]': 'isHiddenByGroup() ? "true" : null',
  },
  template: `
    <a
      [attr.href]="href() || null"
      [attr.target]="href() ? (target() || '_self') : null"
      [attr.role]="href() ? null : 'button'"
      [attr.tabindex]="disabled() ? -1 : (href() ? null : 0)"
      [attr.aria-disabled]="disabled() ? 'true' : null"
      [attr.aria-label]="effectiveLabel() || null"
      [matTooltip]="effectiveLabel()"
      matTooltipPosition="right"
      [matTooltipShowDelay]="150"
      [matTooltipDisabled]="!isDrawerCollapsed()"
      (keydown)="onKeydown($event)"
      class="nav-link">
      @if (icon()) {
        <mat-icon
          fontSet="material-symbols-outlined"
          class="nav-icon material-symbols-outlined"
          aria-hidden="true">{{ icon() }}</mat-icon>
      }
      <span #navText class="nav-text"><ng-content></ng-content></span>
      <span class="nav-badge-slot">
        <ng-content select="[slot=badge], [badge], rack-badge, rack-status-tag, [slot=trailing], [trailing]"></ng-content>
        @if (badge()) {
          <span class="nav-badge" [attr.data-variant]="badgeVariant()">
            {{ badge() }}
          </span>
        }
      </span>
    </a>

    @if (!isDrawerCollapsed()) {
      <button
        type="button"
        class="pin-btn"
        tabindex="-1"
        [disabled]="disabled()"
        [class.pinned]="pinned()"
        (click)="togglePin($event)"
        [attr.aria-label]="pinned() ? 'Unpin item' : 'Pin item'">
        <mat-icon
          fontSet="material-symbols-outlined"
          class="pin-icon material-symbols-outlined"
          [class.filled]="pinned()"
          aria-hidden="true">
          push_pin
        </mat-icon>
      </button>
    }
  `,
})
export class RackNavigationItem implements AfterViewInit {
  readonly icon = input<string>('');
  readonly label = input<string>('');
  readonly active = model<boolean>(false);
  readonly pinned = model<boolean>(false);
  readonly disabled = input<boolean, unknown>(false, {
    transform: booleanAttribute,
  });
  readonly badge = input<string>('');
  readonly badgeVariant = input<string>('primary');
  readonly href = input<string>('');
  readonly target = input<string>('');

  readonly pinnedChangeKebab = output<boolean>({alias: 'pinned-change'});

  private readonly parentGroup = inject(NAV_GROUP_CONTEXT, {optional: true});
  private readonly parentDrawer = inject(NAV_DRAWER_CONTEXT, {optional: true});

  readonly navTextRef = viewChild<ElementRef<HTMLElement>>('navText');
  private readonly projectedLabel = signal<string>('');
  readonly effectiveLabel = computed(
    () => this.label() || this.projectedLabel(),
  );

  readonly isDrawerCollapsed = computed(() => {
    return this.parentDrawer?.collapsed() ?? false;
  });

  readonly isHiddenByGroup = computed(() => {
    if (!this.parentGroup) return false;
    return !this.parentGroup.expanded() && !this.pinned();
  });

  ngAfterViewInit() {
    const text = this.navTextRef()?.nativeElement.textContent?.trim() ?? '';
    if (text) {
      this.projectedLabel.set(text);
    }
  }

  togglePin(event: Event) {
    event.stopPropagation();
    const next = !this.pinned();
    this.pinned.set(next);
    this.pinnedChangeKebab.emit(next);
  }

  onKeydown(event: KeyboardEvent) {
    if (!this.href() && (event.key === ' ' || event.key === 'Enter')) {
      event.preventDefault();
      (event.currentTarget as HTMLElement).click();
    }
  }
}
