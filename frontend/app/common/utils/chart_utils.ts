/** Utility functions for chart DataTables and visual rendering. */

/**
 * Clamps numeric values in a DataTable to 0 if they are non-finite or negative.
 *
 * @param dataTable The DataTable whose numeric values should be clamped.
 * @param startCol The column index to start clamping from (defaults to 0).
 */
export function clampDataTableNumericValues(
  dataTable: google.visualization.DataTable | null | undefined,
  startCol = 0,
): void {
  if (!dataTable) {
    return;
  }
  const rowCount = dataTable.getNumberOfRows();
  const colCount = dataTable.getNumberOfColumns();
  for (let r = 0; r < rowCount; r++) {
    for (let c = startCol; c < colCount; c++) {
      const val = dataTable.getValue(r, c);
      if (typeof val === 'number' && (!Number.isFinite(val) || val < 0)) {
        dataTable.setValue(r, c, 0);
      }
    }
  }
}
