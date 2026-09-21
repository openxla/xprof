import {ChangeDetectionStrategy, Component, effect, input} from '@angular/core';
import {MatOption} from '@angular/material/core';
import {MatDivider} from '@angular/material/divider';
import {MatFormField} from '@angular/material/form-field';
import {MatSelect} from '@angular/material/select';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {
  HostOpTable,
  type MetaHostOpTable,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {Chart} from '../../chart/chart';

/** A host-op view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'host-op',
  templateUrl: './host_op.ng.html',
  styleUrls: ['./host_op.scss'],
  imports: [Chart, MatDivider, MatFormField, MatOption, MatSelect],
})
export class HostOp {
  /** Whether there are host-op tables */
  readonly hasHostOpTables = input(false);

  /** The meta host-op table */
  readonly metaHostOpTable = input<MetaHostOpTable | null>(null);

  /** Array of host-op tables */
  readonly hostOpTables = input<HostOpTable[]>([]);

  allHostOpChoices: string[] = [];

  allHostnameChoices: string[] = [];

  allCoreChoices: string[] = [];

  hostOpSelected: string = '';

  hostnameSelected: string = '';

  coreSelected: string = '';

  showChart = false;

  options: google.visualization.ScatterChartOptions = {
    hAxis: {title: 'TPU step number'},
    vAxis: {title: 'Host-op step number - TPU step number'},
    chartArea: {
      width: '60%',
      height: '60%',
    },
    width: 1000,
    height: 400,
  };

  dataInfo: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
    options: this.options,
  };

  constructor() {
    effect(() => {
      this.setupChoicesAndCharts();
    });
  }

  /** Updates the visability of all charts */
  private updateChartsVisability() {
    const hostOpTables = this.hostOpTables();
    for (let i = 0; i < hostOpTables.length; i++) {
      const prop = hostOpTables[i].p;
      const hostOp = (prop && prop.hostop) || '';
      const hostname = (prop && prop.hostname) || '';
      const core = (prop && prop.value) || '';

      const hostOpMatched: boolean =
        this.hostOpSelected === 'All-ops-in-separate-graph' ||
        this.hostOpSelected === hostOp;
      const hostnameMatched: boolean = this.hostnameSelected === hostname;
      const coreMatched: boolean = this.coreSelected === core;

      if (hostOpMatched && hostnameMatched && coreMatched) {
        this.options.title = this.hostOpChartTitle(hostOp, hostname, core);
        this.dataInfo = {
          ...this.dataInfo,
          data: hostOpTables[i],
        };
        this.showChart = true;
        return;
      }
    }

    this.showChart = false;
  }

  updateHostOpSelection(selection: string) {
    this.hostOpSelected = selection;
    this.updateChartsVisability();
  }

  updateHostnameSelection(selection: string) {
    this.hostnameSelected = selection;
    this.updateChartsVisability();
  }

  updateCoreSelection(selection: string) {
    this.coreSelected = selection;
    this.updateChartsVisability();
  }

  /** Returns the title of a host-op chart */
  private hostOpChartTitle(hostOpName: string, hostname: string, core: string) {
    let hostnameTitle = '';
    if (hostname === 'All-hosts-in-one-graph') {
      hostnameTitle = 'All';
    } else {
      hostnameTitle = hostname;
    }
    let coreTitle = '';
    if (core === 'All-cores') {
      coreTitle = 'All';
    } else {
      coreTitle = core;
    }
    return (
      'Op: ' +
      hostOpName +
      ', Hostname:' +
      hostnameTitle +
      ', Core:' +
      coreTitle
    );
  }

  /** Sets up choices and charts */
  private setupChoicesAndCharts() {
    const metaHostOpTable = this.metaHostOpTable();
    if (!metaHostOpTable) return;

    // Sets up choices.
    this.allHostOpChoices = (metaHostOpTable.p?.valid_host_ops || '').split(
      ',',
    );
    this.allHostnameChoices = (metaHostOpTable.p?.hostnames || '').split(',');
    this.allCoreChoices = (metaHostOpTable.p?.values || '').split(',');

    // Updates visability.
    this.updateChartsVisability();
  }
}
