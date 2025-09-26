import {Component, inject, Input} from '@angular/core';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';

/**
 * A 'View Architecture' button component which currently generates a graphviz
 * URL for the device (TPU/GPU) utilization viewer based on the used device
 * architecture in the program code.
 */
@Component({
  standalone: false,
  selector: 'view-architecture',
  templateUrl: './view_architecture.ng.html',
  styleUrls: ['./view_architecture.scss'],
})
export class ViewArchitecture {
  @Input() sessionId = '';

  dataService: DataServiceV2Interface = inject(DATA_SERVICE_INTERFACE_TOKEN);

  viewArchitecture() {
    this.dataService.generateGraphvizUrlForUtilizationViewer(this.sessionId);
  }
}
