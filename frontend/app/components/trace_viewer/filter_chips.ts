import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  input,
  output,
  viewChild,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatAutocompleteModule} from '@angular/material/autocomplete';
import {MatButtonModule} from '@angular/material/button';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatChipEditedEvent, MatChipsModule} from '@angular/material/chips';
import {MatIconModule} from '@angular/material/icon';
import {MatTooltipModule} from '@angular/material/tooltip';

import {
  MatAutocomplete,
  MatAutocompleteTrigger,
} from '@angular/material/autocomplete';
import {BehaviorSubject} from 'rxjs';
import {
  FilterFieldCategory,
  FilterOperatorType,
  type FilterChangeEvent,
  type FilterEntry,
  type FilterRemoveEvent,
  type FilterValue,
} from './trace_viewer_typings';

const CHIP_TEXT_MAX_LENGTH = 15;

/**
 * Component to display a list of selected filter chips
 */
@Component({
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    FormsModule,
    MatAutocompleteModule,
    MatButtonModule,
    MatCheckboxModule,
    MatChipsModule,
    MatIconModule,
    MatTooltipModule,
  ],
  selector: 'filter-chips',
  template: `
    <mat-chip-grid #chipGrid>
      <ng-container *ngFor="let filter of filters(); let idx = index">
        <mat-chip-row
          matAutocompleteOrigin #origin="matAutocompleteOrigin"
          (removed)="remove(idx)"
          [editable]="true"
          (edited)="edit(idx, $event)"
          [value]="filter.value"
          [attr.aria-description]="'press enter to edit filter ' + filter.field.displayName"
          [matTooltip]="getTooltip(filter)"
          (click)="onClickChip($event, filter, idx)"
          class="filter-chip">
            {{getFilterShortenString(filter)}}
          <button matChipRemove [attr.aria-label]="'remove filter ' + filter.field.displayName">
            <mat-icon>cancel</mat-icon>
          </button>
          <input
            hidden
            #optionTrigger="matAutocompleteTrigger"
            [matAutocompleteConnectedTo]="origin"
            [matAutocomplete]="chipValueOptionsAuto"
            [attr.aria-label]="'Filter options for ' + filter.field.displayName"
            [id]="'chip-dummy-input-' + idx"
            class="chip-dummy-input" />
        </mat-chip-row>
        <mat-autocomplete #chipValueOptionsAuto class="dense" panelWidth="fit-content">
          <div style="display:flex;flex-direction:column;">
            <button mat-stroked-button color="primary" (click)="onChipMultiSelectUpdateConfirm()" style="margin:10px;">Confirm</button>
            <mat-option>
              <mat-checkbox class="example-margin" [checked]="allOptionsSelected" (click)="onOperateAll($event)">{{allOptionsLabel}}</mat-checkbox>
            </mat-option>
            <mat-option *ngFor="let option of (autoChipValueOptions | async) trackBy:trackByValue"
              [value]="option.value" >
              <mat-checkbox class="example-margin" [(ngModel)]="option.checked" (click)="onClickChipOption($event)">{{option.value}}</mat-checkbox>
            </mat-option>
          </div>
        </mat-autocomplete>
      </ng-container>
    </mat-chip-grid>
`,
  styleUrls: ['./trace_viewer.scss'],
})
export class FilterChips {
  readonly filters = input<FilterEntry[]>([]);
  readonly hosts = input<string[]>([]);
  readonly processes = input<string[]>([]);

  readonly filterChanged = output<FilterChangeEvent>();
  readonly filterRemoved = output<FilterRemoveEvent>();
  readonly chipValueOptionsAuto = viewChild.required<MatAutocomplete>(
    'chipValueOptionsAuto',
  );
  readonly optionTrigger = viewChild<MatAutocompleteTrigger>('optionTrigger');

  autoChipValueOptions = new BehaviorSubject<FilterValue[]>([]);
  onEditChipIndex = -1;

  get allOptionsSelected() {
    return this.autoChipValueOptions.value.every((option) => option.checked);
  }

  get allOptionsLabel() {
    return this.allOptionsSelected
      ? 'Deselect All'
      : 'Select All Displayed Options';
  }

  onOperateAll(e: Event) {
    e.stopPropagation();
    const allOptionsSelected = this.allOptionsSelected;
    this.autoChipValueOptions.value.forEach((option) => {
      option.checked = !allOptionsSelected;
    });
  }

  trackByValue(index: number, option: FilterValue): string {
    return option.value || '';
  }

  onClickChip(e: Event, filter: FilterEntry, index: number) {
    e.stopPropagation();
    if (this.optionTrigger()?.panelOpen) {
      this.onEditChipIndex = -1;
      this.optionTrigger()?.closePanel();
    } else {
      this.onEditChipIndex = index;
      const options = this.getChipOptions(filter);
      if (options.length > 0) {
        this.autoChipValueOptions.next(options);
        this.optionTrigger()?.openPanel();
      }
    }
  }

  onClickChipOption(e: Event) {
    e.stopPropagation();
  }

  getChipOptions(filter: FilterEntry) {
    if (
      filter.field.hasMultiSelectOptions &&
      filter.operator.value === FilterOperatorType.EXACT
    ) {
      if (filter.field.info.category === FilterFieldCategory.HOST) {
        return this.hosts().map((host) => {
          return {value: host, checked: filter.value.split(',').includes(host)};
        });
      } else if (filter.field.info.category === FilterFieldCategory.PROCESS) {
        return this.processes().map((process) => {
          return {value: process, checked: filter.value.includes(process)};
        });
      }
    }
    return [];
  }

  onChipMultiSelectUpdateConfirm() {
    const updatedChipValue = this.autoChipValueOptions.value
      .filter((option) => option.checked)
      .map((option) => option.value)
      .join(',');
    if (this.onEditChipIndex >= 0) {
      this.filterChanged.emit({
        value: updatedChipValue,
        index: this.onEditChipIndex,
      });
    }
    this.optionTrigger()?.closePanel();
    this.onEditChipIndex = -1;
  }

  remove(index: number) {
    this.filterRemoved.emit({index});
  }

  edit(index: number, event: MatChipEditedEvent) {
    this.filterChanged.emit({value: event.value, index});
  }

  getTooltip(filter: FilterEntry) {
    return `${filter.field?.displayName}${filter.operator.value}${filter.value}`;
  }

  getFilterShortenString(filter: FilterEntry) {
    const filterString = this.getTooltip(filter);
    return filterString.length <= CHIP_TEXT_MAX_LENGTH
      ? filterString
      : filterString.substring(0, CHIP_TEXT_MAX_LENGTH) + '...';
  }
}
