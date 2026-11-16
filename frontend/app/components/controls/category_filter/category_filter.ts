import {Component, EventEmitter, Input, OnChanges, Output, SimpleChanges, ChangeDetectionStrategy} from '@angular/core';

/**
 * A category filter component.
 * The options are all unique values in the given column of dataTable.
 * Each cell in the column contains one value or multiple values if a
 * valueSeparator is given.
 * Selects the rows that contain the value selected by the user.
 * If all is set, it is used as a value that selects all rows.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'category-filter',
  templateUrl: './category_filter.ng.html',
  styleUrls: ['./category_filter.scss']
})
export class CategoryFilter implements OnChanges {
  @Input() dataTable?: google.visualization.DataTable;
  @Input() column: number|string = -1;
  @Input() valueSeparator = '';
  @Input() all = '';
  @Input() initValue: number|string|boolean = '';

  columnIndex = -1;
  columnLabel = '';
  options: Array<number|string|boolean> = [];
  value: number|string|boolean = '';

  @Output()
  changed = new EventEmitter<google.visualization.DataTableCellFilter>();

  ngOnChanges(changes: SimpleChanges) {
    this.processData();
  }

  processData() {
    if (!this.dataTable || this.dataTable.getNumberOfRows() === 0) {
      return;
    }

    this.columnIndex = this.dataTable.getColumnIndex(this.column);
    if (this.columnIndex !== -1) {
      this.columnLabel = this.dataTable.getColumnLabel(this.columnIndex);
      const values = new Set<number|string|boolean>();
      const seenLower = new Set<string>();
      const numRows = this.dataTable.getNumberOfRows();
      for (let i = 0; i < numRows; ++i) {
        const v = this.dataTable.getValue(i, this.columnIndex);
        if (v == null) continue;
        if (this.valueSeparator) {
          for (const value of String(v).split(this.valueSeparator)) {
            if (!value) continue;
            const lower = value.toLowerCase();
            if (!seenLower.has(lower)) {
              seenLower.add(lower);
              values.add(value);
            }
          }
        } else {
          values.add(v);
        }
      }
      this.options = Array.from(values);
      if (this.dataTable.getColumnType(this.columnIndex) === 'number') {
        this.options.sort((a, b) => (a as number) - (b as number));
      } else {
        // Same order as the default sort, which compares string forms.
        this.options.sort((a, b) => {
          const left = String(a);
          const right = String(b);
          return left < right ? -1 : left > right ? 1 : 0;
        });
      }
      if (this.all) {
        this.options.unshift(this.all);
      }
      const hasInitValue = this.initValue !== '' && this.initValue != null;
      if (hasInitValue && this.valueSeparator) {
        // Separator options are deduped ignoring case, so match the initial
        // value the same way. Single-value cells keep exact matching.
        const initValue = String(this.initValue).toLowerCase();
        this.value =
            this.options.find(
                (option) => String(option).toLowerCase() === initValue) ??
            this.initValue;
      } else if (hasInitValue) {
        this.value = this.initValue;
      } else {
        this.value = this.options[0];
      }
    }

    this.updateFilter();
  }

  updateFilter() {
    if (!this.dataTable || this.dataTable.getNumberOfRows() === 0) {
      return;
    }

    const filter:
        google.visualization.DataTableCellFilter = {column: this.columnIndex};
    if (!this.all || this.value !== this.all) {
      if (this.valueSeparator) {
        const separator = this.valueSeparator;
        const target = String(this.value ?? '').toLowerCase();
        filter.test = (value: unknown): boolean =>
            value != null &&
            target !== '' &&
            String(value)
                .split(separator)
                .some((item) => item.toLowerCase() === target);
      } else {
        filter.value = this.value;
      }
    }

    this.changed.emit(filter);
  }
}
