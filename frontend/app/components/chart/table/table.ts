import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  OnInit,
  effect,
  input,
  viewChild,
} from '@angular/core';
import {MatOption} from '@angular/material/core';
import {MatFormField, MatLabel} from '@angular/material/form-field';
import {MatSelect} from '@angular/material/select';

/** A table view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'table',
  templateUrl: './table.ng.html',
  styleUrls: ['./table.scss'],
  imports: [MatFormField, MatLabel, MatOption, MatSelect],
})
export class Table implements OnInit {
  readonly dataView = input<google.visualization.DataView>();
  readonly showRowNumber = input(false);
  readonly page = input('disable');
  readonly pageSizeOptions = input<number[]>([]);
  readonly pageSize = input<number, number | string>(10, {
    transform: (value: number | string) =>
      typeof value === 'string' ? Number(value) : value,
  });

  table?: google.visualization.Table;
  height = '150px';
  selectedPageSize = 10;

  readonly tableRef = viewChild<ElementRef>('table');

  constructor() {
    effect(() => {
      const options = this.pageSizeOptions();
      if (options.length > 0) {
        this.selectedPageSize = options[0];
      } else {
        this.selectedPageSize = this.pageSize();
      }
      this.drawTable();
    });
  }

  ngOnInit() {
    this.loadGoogleChart();
    this.populateDefaultPageSize();
  }

  @HostListener('window:resize')
  onResize() {
    const tableElement = this.tableRef()?.nativeElement.querySelector('table');
    if (tableElement) {
      this.height = String(Number(tableElement.clientHeight) + 20) + 'px';
    }
  }

  drawTable() {
    const dataView = this.dataView();
    if (!this.table || !dataView) {
      return;
    }

    const options: google.visualization.TableOptions = {
      allowHtml: true,
      alternatingRowStyle: false,
      showRowNumber: this.showRowNumber(),
      page: this.page(),
      pageSize: this.selectedPageSize,
      cssClassNames: {
        'headerCell': 'google-chart-table-header-cell',
        'tableCell': 'google-chart-table-table-cell',
      },
    };

    this.table.draw(dataView, options);

    this.onResize();
  }

  displayPageSizeSelector() {
    return this.pageSizeOptions().length > 0;
  }

  populateDefaultPageSize() {
    // when passing pageSizeOptions from parent, pageSize will by default be the
    // 1st element in the list
    if (this.pageSizeOptions().length > 0) {
      this.selectedPageSize = this.pageSizeOptions()[0];
    } else {
      this.selectedPageSize = this.pageSize();
    }
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.safeLoad({'packages': ['table']});
    google.charts.setOnLoadCallback(() => {
      const tableEl = this.tableRef()?.nativeElement;
      if (!tableEl) return;
      this.table = new google.visualization.Table(tableEl);
      this.drawTable();
    });
  }
}
