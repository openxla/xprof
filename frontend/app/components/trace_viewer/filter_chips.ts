import {
  ChangeDetectionStrategy,
  Component,
  input,
  output,
  viewChild,
} from '@angular/core';
import {
  MatChipEditedEvent,
  MatChipGrid,
  MatChipRemove,
  MatChipRow,
} from '@angular/material/chips';

import {AsyncPipe} from '@angular/common';
import {FormsModule} from '@angular/forms';
import {
  MatAutocomplete,
  MatAutocompleteOrigin,
  MatAutocompleteTrigger,
} from '@angular/material/autocomplete';
import {MatButton} from '@angular/material/button';
import {MatCheckbox} from '@angular/material/checkbox';
import {MatOption} from '@angular/material/core';
import {MatIcon} from '@angular/material/icon';
import {MatTooltip} from '@angular/material/tooltip';
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
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'filter-chips',
  templateUrl: './filter_chips.ng.html',
  styleUrls: ['./trace_viewer.scss'],
  imports: [
    AsyncPipe,
    FormsModule,
    MatAutocomplete,
    MatAutocompleteOrigin,
    MatAutocompleteTrigger,
    MatButton,
    MatCheckbox,
    MatChipGrid,
    MatChipRemove,
    MatChipRow,
    MatIcon,
    MatOption,
    MatTooltip,
  ],
})
export class FilterChips {
  readonly filters = input<FilterEntry[]>([]);
  readonly hosts = input<string[]>([]);
  readonly processes = input<string[]>([]);

  readonly filterChanged = output<FilterChangeEvent>();
  readonly filterRemoved = output<FilterRemoveEvent>();
  readonly chipValueOptionsAuto = viewChild<MatAutocomplete>(
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
