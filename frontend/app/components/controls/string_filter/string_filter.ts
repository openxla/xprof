import {
  ChangeDetectionStrategy,
  Component,
  effect,
  input,
  output,
} from '@angular/core';
import {MatIconButton} from '@angular/material/button';
import {MatFormField, MatLabel, MatSuffix} from '@angular/material/form-field';
import {MatIcon} from '@angular/material/icon';
import {MatInput} from '@angular/material/input';
import {MatTooltip} from '@angular/material/tooltip';

/**
 * A string filter component.
 * Selects the rows that contain the value typed by the user.
 * If the value is empty, selects all the rows.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'string-filter',
  templateUrl: './string_filter.ng.html',
  styleUrls: ['./string_filter.scss'],
  imports: [
    MatFormField,
    MatIcon,
    MatIconButton,
    MatInput,
    MatLabel,
    MatSuffix,
    MatTooltip,
  ],
})
export class StringFilter {
  readonly dataTable = input<google.visualization.DataTable>();
  readonly column = input<number | string>(-1);
  readonly valueInput = input('', {alias: 'value'});
  readonly exactMatchInput = input(false, {alias: 'exactMatch'});
  readonly matchToggle = input(false);

  filterValue = '';
  exactMatch = false;
  columnIndex = -1;
  columnLabel = '';

  readonly changed = output<google.visualization.DataTableCellFilter>();

  constructor() {
    effect(() => {
      this.filterValue = this.valueInput();
      this.exactMatch = this.exactMatchInput();
      this.processData();
    });
  }

  toggleExactMatch() {
    this.exactMatch = !this.exactMatch;
    this.updateFilter();
  }

  onInputChange(val: string) {
    this.filterValue = val;
    this.updateFilter();
  }

  processData() {
    const dataTable = this.dataTable();
    if (!dataTable || dataTable.getNumberOfRows() === 0) {
      return;
    }

    this.columnIndex = dataTable.getColumnIndex(this.column());
    if (this.columnIndex !== -1) {
      this.columnLabel = dataTable.getColumnLabel(this.columnIndex);
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
    if (this.filterValue) {
      if (this.exactMatch) {
        filter.test = (value: string) =>
          value.toLowerCase().trim() === this.filterValue.toLowerCase().trim();
      } else {
        filter.test = (value: string) =>
          value.toLowerCase().indexOf(this.filterValue.toLowerCase()) !== -1;
      }
    }

    this.changed.emit(filter);
  }
}
