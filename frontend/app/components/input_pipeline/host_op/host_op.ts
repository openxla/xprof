import {
  ChangeDetectionStrategy,
  Component,
  Input,
  OnChanges,
  OnInit,
  SimpleChanges,
} from '@angular/core';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {
  HostOpTable,
  type MetaHostOpTable,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';

/** A host-op view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'host-op',
  templateUrl: './host_op.ng.html',
  styleUrls: ['./host_op.scss'],
})
export class HostOp implements OnChanges, OnInit {
  /** Whether there are host-op tables */
  @Input() hasHostOpTables = false;

  /** The meta host-op table */
  @Input() metaHostOpTable: MetaHostOpTable | null = null;

  /** Array of host-op tables */
  @Input() hostOpTables: HostOpTable[] = [];

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

  /** Updates the visability of all charts */
  private updateChartsVisability() {
    for (let i = 0; i < this.hostOpTables.length; i++) {
      const prop = this.hostOpTables[i]?.p;
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
          data: this.hostOpTables[i],
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
  private setupChoicesAndCharts(resetSelections = false) {
    if (!this.metaHostOpTable) {
      this.allHostOpChoices = [];
      this.allHostnameChoices = [];
      this.allCoreChoices = [];
      this.hostOpSelected = '';
      this.hostnameSelected = '';
      this.coreSelected = '';
      this.showChart = false;
      this.dataInfo = {
        ...this.dataInfo,
        data: null,
      };
      return;
    }

    // Sets up choices.
    this.allHostOpChoices = (
      this.metaHostOpTable.p?.valid_host_ops || ''
    )
      .split(',')
      .filter((choice) => choice !== '');
    this.allHostnameChoices = (this.metaHostOpTable.p?.hostnames || '')
      .split(',')
      .filter((choice) => choice !== '');
    this.allCoreChoices = (this.metaHostOpTable.p?.values || '')
      .split(',')
      .filter((choice) => choice !== '');

    const defaultTableProps = this.hostOpTables[0]?.p;
    if (
      resetSelections ||
      !this.hostOpSelected ||
      !this.allHostOpChoices.includes(this.hostOpSelected)
    ) {
      const preferredOp = defaultTableProps?.hostop || '';
      this.hostOpSelected = this.allHostOpChoices.includes(preferredOp)
        ? preferredOp
        : this.allHostOpChoices[0] || '';
    }
    if (
      resetSelections ||
      !this.hostnameSelected ||
      !this.allHostnameChoices.includes(this.hostnameSelected)
    ) {
      const preferredHost = defaultTableProps?.hostname || '';
      this.hostnameSelected = this.allHostnameChoices.includes(preferredHost)
        ? preferredHost
        : this.allHostnameChoices[0] || '';
    }
    if (
      resetSelections ||
      !this.coreSelected ||
      !this.allCoreChoices.includes(this.coreSelected)
    ) {
      const preferredCore = defaultTableProps?.value || '';
      this.coreSelected = this.allCoreChoices.includes(preferredCore)
        ? preferredCore
        : this.allCoreChoices[0] || '';
    }

    // Updates visability.
    this.updateChartsVisability();
  }

  ngOnInit() {
    this.setupChoicesAndCharts();
  }

  ngOnChanges(changes: SimpleChanges) {
    const resetSelections = Boolean(
      changes['metaHostOpTable'] || changes['hostOpTables'],
    );
    this.setupChoicesAndCharts(resetSelections);
  }
}
