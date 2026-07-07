import {CommonModule} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChange,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {
  MAT_AUTOCOMPLETE_DEFAULT_OPTIONS,
  MatAutocomplete,
  MatAutocompleteModule,
  MatAutocompleteSelectedEvent,
  MatAutocompleteTrigger,
} from '@angular/material/autocomplete';
import {MatButtonModule} from '@angular/material/button';
import {MatCheckboxModule} from '@angular/material/checkbox';

import {BehaviorSubject} from 'rxjs';
import {FILTER_FIELDS, FILTER_OPERATORS} from './constants';
import {
  FilterEntry,
  FilterField,
  FilterFieldCategory,
  FilterOperatorType,
  FilterOption,
  FilterValue,
} from './trace_viewer_typings';
import {filterFieldKey, lookupFilterOperator} from './utils';

/**
 * Component to display input field for adding a new filter.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: true,
  selector: 'filter-input',
  imports: [
    CommonModule,
    FormsModule,
    MatAutocompleteModule,
    MatButtonModule,
    MatCheckboxModule,
  ],
  viewProviders: [
    {
      provide: MAT_AUTOCOMPLETE_DEFAULT_OPTIONS,
      useValue: {},
    },
  ],
  template: `
    <div matAutocompleteOrigin #origin="matAutocompleteOrigin" style="display:flex;">
    <input #inputEl type="text"
                #optionTrigger="matAutocompleteTrigger"
                class="filter-input"
                (input)="onInputChange()"
                [value]="filterInput"
                (keyup.enter)="tryAddFilter()"
                (keyup.escape)="onEscape()"
                [matAutocomplete]="filterOptionsAuto"
                placeholder="Add filter..."
                aria-label="Add filter"/>
    </div>

<mat-autocomplete #filterOptionsAuto class="dense" panelWidth="fit-content" (optionSelected)="onOptionSelected($event)">
  <!-- option list for filter field and operator -->
  <div *ngIf="!isUpdatingValues()">
    <mat-option *ngFor="let option of (autoFilterOptions | async) trackBy:trackByValue"
              [value]="option.value">
      {{option.displayName || option.value}}
    </mat-option>
  </div>
  <!-- option list for filter values -->
  <div *ngIf="isUpdatingValues() && isMultiSelect()" style="display:flex;flex-direction:column;">
    <button mat-stroked-button color="primary" (click)="onConfirmMultiSelect()" style="margin:10px;">Confirm</button>
    <mat-option>
      <mat-checkbox class="example-margin" [checked]="allOptionsSelected" (click)="onOperateAll($event)">{{allOptionsLabel}}</mat-checkbox>
    </mat-option>
    <mat-option *ngFor="let option of (autoFilterValues | async) trackBy:trackByValue"
      [value]="option.value" >
      <mat-checkbox class="example-margin" [(ngModel)]="option.checked" (click)="onClickCheckbox($event)">{{option.displayName || option.value}}</mat-checkbox>
    </mat-option>
  </div>
</mat-autocomplete>
`,
  styles: [
    `
    .filter-input {
      outline: none;
      border: none;
      text-align: left;
      padding: 0;
      margin: 0;
      background: transparent;
      width: 100%;
      height: 28px;
      font-size: 14px;
      color: #5f6368;
    }
  `,
  ],
})
export class FilterInput implements AfterViewInit, OnChanges {
  @ViewChild('inputEl') inputEl!: ElementRef;
  @ViewChild('filterOptionsAuto') filterOptionsAuto!: MatAutocomplete;
  @ViewChild('optionTrigger') optionTrigger!: MatAutocompleteTrigger;

  @Output() readonly filterAdded = new EventEmitter<FilterEntry>();
  @Input() validFilterFields: FilterField[] = [];
  @Input() hosts: string[] = [];
  @Input() processes: string[] = [];

  private filterInputInternal = '';
  /**
   * Steps of creating a filter will control what dialog will be triggered for user inputs.
   * 0: define the filter field (eg. host).
   * 1: define the operator (eg. Exact (=)).
   * 2: define the filter value.
   */
  filterStep = 0;
  // currentFilterField,Operator,Value are all string values,
  // can use lookUp function to find the object.
  currentFilterField = '';
  currentFilterOperator = '';
  currentFilterValue = '';

  autoFilterOptions = new BehaviorSubject<FilterOption[]>(FILTER_FIELDS);
  autoFilterFields = new BehaviorSubject<FilterField[]>(FILTER_FIELDS);

  get autoFilterValues() {
    return new BehaviorSubject<FilterValue[]>(this.autoFilterOptions.value);
  }

  get fieldValueOptions(): {[key: string]: string[]} {
    const valueOptions: {[key: string]: string[]} = {};
    valueOptions[FilterFieldCategory.HOST] = this.hosts;
    valueOptions[FilterFieldCategory.PROCESS] = this.processes;
    return valueOptions;
  }

  // The entire filter input string, that the filter field/operator/value are parsed from
  get filterInput() {
    return this.filterInputInternal;
  }

  set filterInput(input: string) {
    const {field, operator, value} = this.parseFilterInput(input);
    this.currentFilterField = field ? `${this.trackByValue(0, field)}:` : input;
    this.currentFilterOperator = field && operator.length ? operator : '';
    this.currentFilterValue =
      field && operator.length && value.length ? value : '';
    this.filterInputInternal = `${this.currentFilterField}${this.currentFilterOperator}${this.currentFilterValue}`;

    this.updateCurrentFilterStep();
    this.updateFilterOptions(input);
  }

  get allOptionsSelected() {
    return this.autoFilterValues.value.every((option) => option.checked);
  }

  get allOptionsLabel() {
    return this.allOptionsSelected
      ? 'Deselect All'
      : 'Select All Displayed Options';
  }

  ngOnChanges(changes: SimpleChanges) {
    if (this.validFilterFieldsChanged(changes['validFilterFields'])) {
      this.autoFilterFields.next(this.validFilterFields);
      this.updateFilterOptions();
    }
  }

  ngAfterViewInit() {
    if (!(this.optionTrigger instanceof MatAutocompleteTrigger)) {
      throw new Error('@ViewChild "trigger" is required');
    }
  }

  isUpdatingValues() {
    return this.filterStep === 2;
  }

  multiSelectEnabled(filterField: FilterField | undefined) {
    return (
      filterField?.operatorTypes?.includes(FilterOperatorType.EXACT) &&
      filterField?.hasMultiSelectOptions
    );
  }

  isMultiSelect() {
    if (this.currentFilterOperator !== FilterOperatorType.EXACT) return false;
    const filterField = this.lookupFilterField(this.currentFilterField);
    return this.multiSelectEnabled(filterField);
  }

  onConfirmMultiSelect() {
    this.currentFilterValue = this.autoFilterValues.value
      .filter((value) => value.checked)
      .map((value) => value.value)
      .join(',');
    this.filterInput = `${this.currentFilterField}${this.currentFilterOperator}${this.currentFilterValue}`;
    this.tryAddFilter();
  }

  onClickCheckbox(e: Event) {
    e.stopPropagation();
  }

  onOperateAll(e: Event) {
    e.stopPropagation();
    const allOptionsSelected = this.allOptionsSelected;
    this.autoFilterValues.value.forEach((option) => {
      option.checked = !allOptionsSelected;
    });
  }

  trackByValue(index: number, option: FilterOption): string {
    // FilterOption can be FilterField, FilterOperator, FilterValue.
    // FilterField's value variable is an object
    // For other future interfaces that may be added, such as FilterField,
    // we will need to add discrimination logic here and to interface definitions.
    if (typeof option.value === 'object') {
      return filterFieldKey(option as FilterField) || '';
    }
    return option.value || '';
  }

  // Tokenizing the input
  parseFilterInput(input: string) {
    const parts = this.splitOnFirstColon(input);
    const filterFieldValue: string = parts[0];
    const filterField = this.lookupFilterField(filterFieldValue);

    // Operators that are of length 2 will always end with an '='
    // If this changes, this logic will need to be updated.
    const opTokenLength = 1 + (parts[1].slice(0, 2).endsWith('=') ? 1 : 0);

    const filterOperator: string = parts[1].slice(0, opTokenLength);
    // Trim the surrounding space for filter value
    const filterValue: string = parts[1].slice(opTokenLength).trim();
    return {
      field: filterField,
      operator: filterOperator,
      value: filterValue,
    };
  }

  updateFilterOptions(input = '') {
    let currentOptions: FilterOption[] = [];
    switch (this.filterStep) {
      case 0:
        const filteredNameOptions = this.validFilterFields.filter(
          (field: FilterField) =>
            field.displayName
              .toLowerCase()
              .startsWith(input.trim().toLowerCase()),
        );
        currentOptions = filteredNameOptions;
        break;
      case 1:
        const filterField = this.lookupFilterField(this.currentFilterField);
        currentOptions = FILTER_OPERATORS.filter((operator) =>
          filterField?.operatorTypes?.includes(operator.value),
        );
        break;
      case 2:
        // Only multi-select has dropdown for value selection.
        // Otherwise always close the panel and take user input directly.
        if (this.currentFilterOperator === FilterOperatorType.EXACT) {
          const filterField = this.lookupFilterField(this.currentFilterField);
          if (filterField && this.multiSelectEnabled(filterField)) {
            const fieldKey = filterFieldKey(filterField);
            const options = this.fieldValueOptions[fieldKey] || [];
            currentOptions = options
              .map((option) => ({value: option, checked: false}))
              .filter((option) =>
                option.value.includes(this.currentFilterValue),
              );
          }
        } else {
          currentOptions = [];
          this.optionTrigger.closePanel();
        }
        break;
      default:
        currentOptions = [];
        break;
    }
    this.autoFilterOptions.next(currentOptions);
  }

  updateCurrentFilterStep() {
    if (!this.filterInput.includes(':')) {
      this.filterStep = 0;
    } else {
      const {operator} = this.parseFilterInput(this.filterInput);
      if (
        !FILTER_OPERATORS.map((operator) => operator.value).includes(operator)
      ) {
        this.filterStep = 1;
      } else {
        this.filterStep = 2;
      }
    }
  }

  // Will always return an array of length 2
  private splitOnFirstColon(s: string): string[] {
    const fieldPart = s.split(':')[0].trim();
    const valuePart = s.split(':').slice(1).join(':').trim();
    return [fieldPart, valuePart];
  }

  // Returns filter name if the string matches one of the display names
  // Otherwise returns undefined.
  private lookupFilterField(fieldValue: string): FilterField | undefined {
    return this.validFilterFields.find(
      (filterField: FilterField) =>
        filterFieldKey(filterField).toLowerCase() ===
          fieldValue.trim().toLowerCase() ||
        `${filterFieldKey(filterField).toLowerCase()}:` ===
          fieldValue.trim().toLowerCase(),
    );
  }

  private validFilterFieldsChanged(change: SimpleChange) {
    const fieldsStrRepresentation = (fields: FilterField[]) =>
      fields.map((field: FilterField) => field.displayName).join(',');
    return (
      fieldsStrRepresentation(change?.currentValue || []) !==
      fieldsStrRepresentation(change?.previousValue || [])
    );
  }

  onOptionSelected(event: MatAutocompleteSelectedEvent) {
    const selectedValue = this.trackByValue(0, event.option);
    switch (this.filterStep) {
      case 0:
        this.filterInput = selectedValue;
        this.focus();
        break;
      case 1:
        this.filterInput = `${this.currentFilterField}${selectedValue}`;
        this.focus();
        break;
      case 2:
        this.filterInput = `${this.currentFilterField}${this.currentFilterOperator}${selectedValue}`;
        break;
      default:
    }
  }

  focus() {
    setTimeout(() => {
      this.inputEl.nativeElement.focus();
      this.optionTrigger?.openPanel();
    }, 100);
  }

  onEscape() {
    switch (this.filterStep) {
      case 0:
        this.reset();
        break;
      case 1:
        this.currentFilterOperator = '';
        break;
      case 2:
        this.currentFilterValue = '';
        break;
      default:
        break;
    }
  }

  onInputChange() {
    this.filterInput = this.inputEl.nativeElement.value;
  }

  isValidOperator(field: FilterField, operator: string) {
    const fieldInfo = filterFieldKey(field);
    const filterField = this.lookupFilterField(fieldInfo);
    return operator.length && filterField?.operatorTypes?.includes(operator);
  }

  tryAddFilter(): boolean {
    const {field, operator, value} = this.parseFilterInput(this.filterInput);
    if (this.filterStep < 2) {
      return false;
    }

    if (field && this.isValidOperator(field, operator) && value.length) {
      const filter: FilterEntry = {
        field,
        value,
        operator: lookupFilterOperator(operator),
      };
      this.filterAdded.next(filter);
      this.reset();
      return true;
    }
    return false;
  }

  private reset() {
    this.filterInput = '';
    this.filterStep = 0;
    this.optionTrigger?.closePanel();
  }
}
