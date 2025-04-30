import {Component, inject, Input} from '@angular/core';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';

/**
 * A 'Export as CSV' button component.
 */
@Component({
  standalone: false,
  selector: 'export-as-csv',
  templateUrl: './export_as_csv.ng.html',
  styleUrls: ['./export_as_csv.scss'],
})
export class ExportAsCsv {
  @Input() tool = '';
  @Input() sessionId = '';
  @Input() host = '';

  dataService: DataServiceV2Interface = inject(DATA_SERVICE_INTERFACE_TOKEN);

  exportDataAsCSV() {
    this.dataService.exportDataAsCSV(this.sessionId, this.tool, this.host);
  }
}
