import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
} from '@angular/core';

/**
 * A string filter component.
 * Selects the rows that contain the value typed by the user.
 * If the value is empty, selects all the rows.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'string-filter',
  templateUrl: './string_filter.ng.html',
  styleUrls: ['./string_filter.scss'],
})
export class StringFilter implements OnChanges {
  @Input() dataTable?: google.visualization.DataTable;
  @Input() column: number | string = -1;
  @Input() value = '';
  @Input() exactMatch = false;
  @Input() matchToggle = false;

  columnIndex = -1;
  columnLabel = '';

  @Output()
  changed = new EventEmitter<google.visualization.DataTableCellFilter>();

  toggleExactMatch() {
    this.exactMatch = !this.exactMatch;
    this.updateFilter();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.processData();
  }

  processData() {
    if (!this.dataTable || this.dataTable.getNumberOfRows() === 0) {
      this.columnIndex = -1;
      this.columnLabel = '';
      return;
    }

    this.columnIndex = this.dataTable.getColumnIndex(this.column);
    if (this.columnIndex === -1) {
      this.columnLabel = '';
      return;
    }

    this.columnLabel = this.dataTable.getColumnLabel(this.columnIndex);
    this.updateFilter();
  }

  updateFilter() {
    if (
      !this.dataTable ||
      this.dataTable.getNumberOfRows() === 0 ||
      this.columnIndex < 0
    ) {
      return;
    }

    const filter: google.visualization.DataTableCellFilter = {
      column: this.columnIndex,
    };
    if (this.value) {
      const query = this.value.toLowerCase();
      const trimmedQuery = query.trim();
      const exactMatch = this.exactMatch;
      filter.test = (value: unknown): boolean => {
        if (value == null) {
          return false;
        }
        const text = String(value).toLowerCase();
        return exactMatch ? text.trim() === trimmedQuery : text.includes(query);
      };
    }

    this.changed.emit(filter);
  }
}
