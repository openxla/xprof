import {DataTableCellValue, SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';

class DataTableForTesting {
  // Note the constructor takes a js object (not a literal string)
  constructor(public data: SimpleDataTable = {cols: [], rows: [], p: {}}) {}
  addColumn(type: string, label?: string, id?: string) {
    this.data?.cols?.push({type, label, id});
  }
  clone() {
    return new DataTableForTesting(this.data);
  }
  getColumnIndex(column: number|string): number {
    if (!this.data || !this.data.cols) {
      return -1;
    }

    for (let i = 0; i < this.data.cols.length; i++) {
      if (this.data.cols[i].id === column ||
          this.data.cols[i].label === column) {
        return i;
      }
    }

    return -1;
  }
  getColumnLabel(columnIndex: number): string {
    if (!this.data || !this.data.cols) {
      return '';
    }
    if (columnIndex >= this.data.cols.length) {
      return '';
    }
    return this.data.cols[columnIndex].label as string;
  }
  setColumnLabel(columnIndex: number, label: string) {
    if (!this.data || !this.data.cols || this.data.cols.length <= columnIndex) {
      return;
    }
    this.data.cols[columnIndex].label = label;
  }
  getColumnType(columnIdx: number) {
    if (!this.data || !this.data.cols || this.data.cols.length <= columnIdx) {
      return null;
    }
    return this.data.cols[columnIdx].type;
  }
  getColumnId(columnIdx: number) {
    if (!this.data || !this.data.cols || this.data.cols.length <= columnIdx) {
      return '';
    }
    return this.data.cols[columnIdx].id;
  }
  getColumnRange(columnIdx: number) {
    if (!this.data || !this.data.rows) {
      return [Infinity, -Infinity];
    }
    let min = Infinity;
    let max = -Infinity;
    this.data.rows.forEach(row => {
      const val = row.c![columnIdx].v as number;
      min = Math.min(val, min);
      max = Math.max(val, max);
    });
    return [min, max];
  }
  getDistinctValues(columnIdx: number) {
    if (!this.data?.rows) {
      return [];
    }
    const valueSet = new Set();
    this.data.rows.forEach(row => {
      const val = row.c![columnIdx].v;
      valueSet.add(val);
    });
    return [...valueSet];
  }
  getNumberOfColumns() {
    if (!this.data || !this.data.cols) {
      return -1;
    }
    return this.data.cols.length;
  }
  getNumberOfRows() {
    if (!this.data || !this.data.rows) {
      return -1;
    }
    return this.data.rows.length;
  }
  getValue(rowIdx: number, columnIdx: number) {
    const numOfRows = this.getNumberOfRows();
    const numofCols = this.getNumberOfColumns();
    if (numOfRows < 0 || numofCols < 0 || rowIdx >= numOfRows ||
        columnIdx >= numofCols) {
      return null;
    }
    return this.data.rows![rowIdx].c![columnIdx]?.v;
  }
  getTableProperty(propName: string): string {
    if (!this.data || !this.data.p) {
      return '';
    }
    return this.data.p[propName];
  }
  getTableProperties(): {[key: string]: string} {
    if (!this.data || !this.data.p) {
      return {};
    }
    return this.data.p;
  }
  getFilteredRows(filters: google.visualization.DataTableCellFilter[]):
      number[] {
    if (!this.data || !this.data.rows) {
      return [];
    }
    if (filters === ([] as AnyDuringTs48Migration)) {
      return [
        ...Array.from<number>({length: this.data.rows.length}).fill(0).keys()
      ];
    }
    const rowsIdxArray: number[] = [];
    this.data.rows.forEach((row, index) => {
      const includeRow = filters.reduce((include, filter) => {
        if (!this.data!.cols || this.data!.cols.length <= filter.column) {
          return include;
        }
        const rowValue = row.c![filter.column].v;
        if (filter.test !== undefined && typeof rowValue === 'string') {
          include = include && filter.test(rowValue);
        } else {
          include = include && (rowValue === filter.value);
        }
        return include;
      }, true);
      if (includeRow) {
        rowsIdxArray.push(index);
      }
    });
    return rowsIdxArray;
  }
  insertColumn() {}
  setColumn() {}
  setValue() {}
  setCell() {}
  addRow(row: google.visualization.DataObjectCell[]|DataTableCellValue[]) {
    this.data?.rows?.push({c: []});
    if (!row) return;
    row.forEach(colVal => {
      if (['number', 'string', 'boolean'].includes(typeof colVal)) {
        this.data?.rows?.slice(-1)[0].c?.push(
            {v: colVal as DataTableCellValue});
      } else {
        this.data?.rows?.slice(-1)[0].c?.push(
            colVal as google.visualization.DataObjectCell);
      }
    });
  }
  addRows() {}
  sort() {}
  getSortedRows(sortColumns: {column: number; desc: boolean}|number|number[]) {
    const sortedRowsIndex: number[] = [];
    // take care of Object case only for now
    if (typeof sortColumns === 'object') {
      const sortContent = this.data?.rows?.reduce((acc, row, index) => {
        acc.push({
          value: row.c![(sortColumns as {
                          column: number;
                          desc: boolean
                        }).column]
                     .v ||
              '',
          rowIndex: index
        });
        return acc;
      }, [] as Array<{value: DataTableCellValue, rowIndex: number}>);
      if (!(sortColumns as {
             column: number;
             desc: boolean
           }).desc) {
        sortContent!.sort((a, b) => a.value < b.value ? 1 : -1);
      } else {
        sortContent!.sort((a, b) => a.value < b.value ? -1 : 1);
      }
      sortContent!.forEach(row => {
        sortedRowsIndex.push(row.rowIndex);
      });
    }
    return sortedRowsIndex;
  }
  setProperty(
      rowIndex: number, columnIndex: number, name: string,
      value: DataTableCellValue) {
    if (!this.data || this.getNumberOfRows() <= rowIndex ||
        this.getNumberOfColumns() <= columnIndex) {
      return;
    }
    this.data.rows![rowIndex].c![columnIndex].p = {
      [name]: value,
      ...(this.data.rows![rowIndex].c![columnIndex].p || {}),
    };
  }
  getProperty(rowIndex: number, columnIndex: number, name: string) {
    if (!this.data || this.getNumberOfRows() <= rowIndex ||
        this.getNumberOfColumns() <= columnIndex ||
        this.data.rows![rowIndex].c![columnIndex].p === undefined) {
      return null;
    }
    return this.data.rows![rowIndex].c![columnIndex].p![name];
  }

  /**
   * Returns a JSON representation of this DataTableForTesting.
   *
   * @return A JSON representation of this DataTableForTesting.
   */
  toJSON(): string {
    return JSON.stringify({
      'cols': this.data?.cols,
      'rows': this.data?.rows,
    });
  }
}

class DataViewForTesting {
  visRows: number[] = [];
  visCols: number[] = [];
  constructor(public table: DataTableForTesting) {
    this.visRows = [
      ...Array.from<number>({length: table.getNumberOfRows()}).fill(0).keys()
    ];
    this.visCols = [
      ...Array.from<number>({length: table.getNumberOfColumns()}).fill(0).keys()
    ];
  }
  getNumberOfColumns() {
    return this.table.getNumberOfColumns();
  }
  getColumnLabel(columnIndex: number) {
    return this.table.getColumnLabel(columnIndex);
  }
  setRows(rowsIdxArray: number[]) {
    this.visRows = rowsIdxArray;
  }
  setColumns() {}
  hideColumns() {}
  getSortedRows(sortColumnIdxes: number[] = []) {
    return [];
  }
  getFilteredRows(filters: google.visualization.DataTableCellFilter[]):
      number[] {
    return [];
  }
  toDataTable() {
    return new DataTableForTesting({
      cols: this.table?.data?.cols?.filter(
                (col: google.visualization.DataObjectColumn, index: number) =>
                    this.visCols.includes(index)) ||
          [],
      rows: this.table?.data?.rows?.filter(
                (row: google.visualization.DataObjectRow, index: number) =>
                    this.visRows.includes(index)) ||
          [],
      p: this.table?.data?.p || {},
    });
  }
}

/** The GViz object for testing */
export const GVIZ_FOR_TESTING = {
  charts: {
    safeLoad: () => {},
    setOnLoadCallback: () => {},
  },
  visualization: {
    AreaChart: () => {
      return {
        draw: () => {},
      };
    },
    arrayToDataTable: () => {
      return new DataTableForTesting();
    },
    NumberFormat: () => {
      return {format: () => {}};
    },
    /* tslint:disable-next-line enforce-name-casing */
    DataTable: (data: SimpleDataTable = {
      cols: [],
      rows: [],
      p: {}
    }) => {
      return new DataTableForTesting(data);
    },
    DataView: (table: DataTableForTesting) => {
      return new DataViewForTesting(table);
    },
    Table: () => {
      return {
        draw: () => {},
      };
    },
    data: {
      group(dt: DataTableForTesting) {
        return dt;
      },
      sum() {},
    }
  },
};
