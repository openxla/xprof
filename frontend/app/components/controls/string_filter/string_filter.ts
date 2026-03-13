import {Component, EventEmitter, Input, OnChanges, Output, SimpleChanges, ChangeDetectionStrategy} from '@angular/core';

/**
 * A string filter component.
 * Selects the rows that contain the value typed by the user.
 * If the value is empty, selects all the rows.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Eager,standalone: false,
  selector: 'string-filter',
  templateUrl: './string_filter.ng.html',
  styleUrls: ['./string_filter.scss'],
})
export class StringFilter implements OnChanges {
  @Input() dataTable?: google.visualization.DataTable;
  @Input() column: number|string = -1;
  @Input() value = '';
  @Input() exactMatch = false;

  columnIndex = -1;
  columnLabel = '';

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
    }

    this.updateFilter();
  }

  updateFilter() {
    if (!this.dataTable || this.dataTable.getNumberOfRows() === 0) {
      return;
    }

    const filter:
        google.visualization.DataTableCellFilter = {column: this.columnIndex};
    if (this.value) {
      if (this.exactMatch) {
        filter.test = (value: string) =>
            value.toLowerCase().trim() === this.value.toLowerCase().trim();
      } else {
        filter.test = (value: string) =>
            value.toLowerCase().indexOf(this.value.toLowerCase()) !== -1;
      }
    }

    this.changed.emit(filter);
  }
}
