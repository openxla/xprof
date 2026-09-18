import {
  ChangeDetectionStrategy,
  Component,
  effect,
  input,
  output,
} from '@angular/core';
import {MatOption} from '@angular/material/core';
import {MatFormField, MatLabel} from '@angular/material/form-field';
import {MatSelect} from '@angular/material/select';

/**
 * A category filter component.
 * The options are all unique values in the given column of dataTable.
 * Each cell in the column contains one value or multiple values if a
 * valueSeparator is given.
 * Selects the rows that contain the value selected by the user.
 * If all is set, it is used as a value that selects all rows.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'category-filter',
  templateUrl: './category_filter.ng.html',
  styleUrls: ['./category_filter.scss'],
  imports: [MatFormField, MatLabel, MatOption, MatSelect],
})
export class CategoryFilter {
  readonly dataTable = input<google.visualization.DataTable>();
  readonly column = input<number | string>(-1);
  readonly valueSeparator = input('');
  readonly all = input('');
  readonly initValue = input<number | string | boolean>('');

  columnIndex = -1;
  columnLabel = '';
  options: Array<number | string | boolean> = [];
  value: number | string | boolean = '';

  readonly changed = output<google.visualization.DataTableCellFilter>();

  constructor() {
    effect(() => {
      this.processData();
    });
  }

  processData() {
    const dataTable = this.dataTable();
    if (!dataTable || dataTable.getNumberOfRows() === 0) {
      return;
    }

    this.columnIndex = dataTable.getColumnIndex(this.column());
    if (this.columnIndex !== -1) {
      this.columnLabel = dataTable.getColumnLabel(this.columnIndex);
      const values = new Set<number | string>();
      const numRows = dataTable.getNumberOfRows();
      for (let i = 0; i < numRows; ++i) {
        const v = dataTable.getValue(i, this.columnIndex);
        if (v === null) continue;
        if (this.valueSeparator()) {
          (v as string).split(this.valueSeparator()).forEach((value) => {
            values.add(value);
          });
        } else {
          values.add(v);
        }
      }
      this.options = Array.from(values);
      if (dataTable.getColumnType(this.columnIndex) === 'number') {
        this.options.sort((a, b) => (a as number) - (b as number));
      } else {
        this.options.sort((a, b) => String(a).localeCompare(String(b)));
      }
      if (this.all()) {
        this.options.unshift(this.all());
      }
      if (this.initValue()) {
        this.value = this.initValue();
      } else {
        this.value = this.options[0];
      }
    }

    this.updateFilter();
  }

  updateFilter() {
    const dataTable = this.dataTable();
    if (!dataTable || dataTable.getNumberOfRows() === 0) {
      return;
    }

    const filter: google.visualization.DataTableCellFilter = {
      column: this.columnIndex,
    };
    if (!this.all() || this.value !== this.all()) {
      if (this.valueSeparator()) {
        filter.test = (value: string) =>
          (this.valueSeparator() + value + this.valueSeparator())
            .toLowerCase()
            .indexOf(
              this.valueSeparator() +
                this.value.toString() +
                this.valueSeparator(),
            ) !== -1;
      } else {
        filter.value = this.value;
      }
    }

    this.changed.emit(filter);
  }
}
