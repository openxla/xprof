import {CommonModule} from '@angular/common';
import {Component, EventEmitter, Input, OnChanges, Output, SimpleChanges} from '@angular/core';
import {MatOptionModule} from '@angular/material/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatSelectModule} from '@angular/material/select';

/**
 * A category filter component.
 * The options are all unique values in the given column of dataTable.
 * Each cell in the column contains one value or multiple values if a
 * valueSeparator is given.
 * Selects the rows that contain the value selected by the user.
 * If all is set, it is used as a value that selects all rows.
 */
@Component({
  standalone: true,
  selector: 'category-filter',
  templateUrl: 'category_filter.ng.html',
  styleUrls: ['category_filter.scss'],
  imports: [
    CommonModule,
    MatOptionModule,
    MatFormFieldModule,
    MatSelectModule,
  ],
})
export class CategoryFilter implements OnChanges {
  @Input() dataTable?: google.visualization.DataTable | null;
  @Input() column: number|string = -1;
  @Input() valueSeparator = '';
  @Input() all = '';
  @Input() initValue: number|string|boolean = '';

  columnIndex = -1;
  columnLabel = '';
  options: Array<number|string|boolean> = [];
  value: number|string|boolean = '';

  @Output()
  readonly changed = new EventEmitter<google.visualization.DataTableCellFilter>();

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
      const values = new Set<number|string>();
      const numRows = this.dataTable.getNumberOfRows();
      for (let i = 0; i < numRows; ++i) {
        const v = this.dataTable.getValue(i, this.columnIndex);
        if (v === null) continue;
        if (this.valueSeparator) {
          (v as string).split(this.valueSeparator).forEach(value => {
            values.add(value);
          });
        } else {
          values.add(v);
        }
      }
      this.options = Array.from(values);
      if (this.dataTable.getColumnType(this.columnIndex) === 'number') {
        this.options.sort((a, b) => (a as number) - (b as number));
      } else {
        this.options.sort();
      }
      if (this.all) {
        this.options.unshift(this.all);
      }
      if (this.initValue) {
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
        filter.test = (value: string) =>
            (this.valueSeparator + value + this.valueSeparator)
                .toLowerCase()
                .indexOf(
                    this.valueSeparator + this.value.toString() +
                    this.valueSeparator) !== -1;
      } else {
        filter.value = this.value;
      }
    }

    this.changed.emit(filter);
  }
}
