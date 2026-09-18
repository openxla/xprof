import {ChangeDetectionStrategy, Component, inject, input} from '@angular/core';
import {MatIcon} from '@angular/material/icon';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';

/**
 * A 'Export as CSV' button component.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'export-as-csv',
  templateUrl: './export_as_csv.ng.html',
  styleUrls: ['./export_as_csv.scss'],
  imports: [MatIcon],
})
export class ExportAsCsv {
  readonly tool = input('');
  readonly sessionId = input('');
  readonly host = input('');
  readonly tqx = input('');
  readonly additionalParams = input<Map<string, string>>(new Map());

  dataService: DataServiceV2Interface = inject(DATA_SERVICE_INTERFACE_TOKEN);

  exportDataAsCSV() {
    this.dataService.exportDataAsCSV(
      this.sessionId(),
      this.tool(),
      this.host(),
      this.tqx(),
      this.additionalParams(),
    );
  }
}
