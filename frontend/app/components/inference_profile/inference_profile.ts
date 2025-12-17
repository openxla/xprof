import {Component, inject, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {InferenceProfileData, InferenceProfileDataProperty, InferenceProfileMetadata, InferenceProfileMetadataProperty, InferenceProfileTable, } from 'org_xprof/frontend/app/common/interfaces/data_table';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** An inference profile component. */
@Component({
  standalone: false,
  selector: 'inference-profile',
  templateUrl: './inference_profile.ng.html',
  styleUrls: ['./inference_profile.css'],
})
export class InferenceProfile implements OnDestroy {
  tool = 'inference_profile';
  sessionId = '';
  host = '';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly throbber = new Throbber(this.tool);
  private readonly dataService = inject(DATA_SERVICE_INTERFACE_TOKEN);

  // All the model IDs and data.
  hasBatching = false;
  hasTensorPattern = false;
  allModelIds: string[] = [];
  allRequestTables: google.visualization.DataTable[] = [];
  allRequestProperties: InferenceProfileDataProperty[] = [];
  allBatchTables: google.visualization.DataTable[] = [];
  allBatchProperties: InferenceProfileDataProperty[] = [];
  allTensorPatternTables: google.visualization.DataTable[] = [];
  allTensorPatternProperties: InferenceProfileDataProperty[] = [];
  // Selected model.
  selectedIndex = 0;
  isInitialLoad = true;
  loading = true;

  requestView?: google.visualization.DataView;
  batchView?: google.visualization.DataView;
  tensorPatternView?: google.visualization.DataView;

  // Names of the columns that can be used to compute percentile.
  requestPercentileColumns: string[] = [];
  batchPercentileColumns: string[] = [];
  // By default, request table and batch table both select the first column
  // "Latency" to compute percentile.
  requestPercentileIndex = 0;
  batchPercentileIndex = 0;

  constructor(
      route: ActivatedRoute,
      private readonly store: Store<{}>,
  ) {
    combineLatest([route.params, route.queryParams])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams]) => {
          this.sessionId = params['sessionId'] || this.sessionId;
          this.processQueryParams(queryParams);
          this.update();
        });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  processQueryParams(queryParams: Params) {
    this.sessionId =
        queryParams['run'] || queryParams['sessionId'] || this.sessionId;
    this.tool = queryParams['tag'] || this.tool;
    this.host = queryParams['host'] || this.host;
  }

  parseMetadata(metadataOrNull: InferenceProfileTable) {
    if (!metadataOrNull) return false;
    const metadata = (metadataOrNull as InferenceProfileMetadata).p as
        InferenceProfileMetadataProperty;
    this.hasBatching = metadata.hasBatching === 'true';
    this.hasTensorPattern = metadata.hasTensorPattern === 'true';

    let parseSuccess = true;
    this.allModelIds = [];
    for (const row of metadataOrNull.rows || []) {
      const modelName = row.c?.[0]?.v;
      if (modelName) {
        this.allModelIds.push(String(modelName));
      } else {
        parseSuccess = false;
      }
    }
    return parseSuccess;
  }

  parseData(data: InferenceProfileTable[]) {
    // <data> is a list of tables.
    // The first table is a metadat table, which contains model ids and whether
    // this inference job is using batching.
    // By default there is one table per model.
    // If batching is enabled, there is one additional table per model for
    // batching related metrics.
    // If tensor pattern is recorded, there is one additional table per model
    // for tensor pattern results.
    if (!data || data.length <= 1) return false;
    if (!this.parseMetadata(data[0])) return false;

    // Check the number of inference tables is correct, and parse table data.
    let expectedNum = 1 + this.allModelIds.length;
    if (this.hasBatching) expectedNum += this.allModelIds.length;
    if (this.hasTensorPattern) expectedNum += this.allModelIds.length;
    if (data.length !== expectedNum) return false;

    for (let i = 1; i < data.length; ) {
      const requestData = data[i] as InferenceProfileData;
      this.allRequestTables.push(
        new google.visualization.DataTable(requestData),
      );
      this.allRequestProperties.push(
        requestData.p as InferenceProfileDataProperty,
      );
      i++;

      if (this.hasBatching) {
        const batchData = data[i] as InferenceProfileData;
        this.allBatchTables.push(new google.visualization.DataTable(batchData));
        this.allBatchProperties.push(
          batchData.p as InferenceProfileDataProperty,
        );
        i++;
      }

      if (this.hasTensorPattern) {
        const tensorPatternData = data[i] as InferenceProfileData;
        this.allTensorPatternTables.push(
          new google.visualization.DataTable(tensorPatternData),
        );
        this.allTensorPatternProperties.push(
          tensorPatternData.p as InferenceProfileDataProperty,
        );
        i++;
      }
    }

    return true;
  }

  updateView() {
    this.requestView = new google.visualization.DataView(
      this.allRequestTables[this.selectedIndex],
    );
    // Percentile, Request ID, Batch ID and Trace Viewer URL are not sortable,
    // all the other columns are sortable.
    this.requestPercentileColumns = [];
    for (let i = 0; i < this.requestView.getNumberOfColumns(); i++) {
      const label = this.requestView.getColumnLabel(i);
      if (
        label !== 'Request ID' &&
        label !== 'Percentile' &&
        label !== 'Trace Viewer URL'
      ) {
        this.requestPercentileColumns.push(label);
      }
    }

    this.batchPercentileColumns = [];
    if (this.allBatchTables.length !== 0) {
      this.batchView = new google.visualization.DataView(
        this.allBatchTables[this.selectedIndex],
      );
      // Percentile, Request ID, Batch ID and Trace Viewer URL are not sortable,
      // all the other columns are sortable.
      for (let i = 0; i < this.batchView.getNumberOfColumns(); i++) {
        const label = this.batchView.getColumnLabel(i);
        if (
          label !== 'Batch ID' &&
          label !== 'Percentile' &&
          label !== 'Trace Viewer URL'
        ) {
          this.batchPercentileColumns.push(label);
        }
      }
    }

    if (this.allTensorPatternTables.length !== 0) {
      this.tensorPatternView = new google.visualization.DataView(
        this.allTensorPatternTables[this.selectedIndex],
      );
    }
  }

  update() {
    if (this.isInitialLoad) {
      setLoadingState(true, this.store, 'Loading inference profile data');
      this.throbber.start();
    }
    // Clear the old data from previous update().
    this.allRequestTables = [];
    this.allBatchTables = [];

    this.dataService
        .getData(
            this.sessionId,
            this.tool,
            this.host,
            new Map([
              [
                'request_column',
                this.requestPercentileColumns[this.requestPercentileIndex],
              ],
              [
                'batch_column',
                this.batchPercentileColumns[this.batchPercentileIndex],
              ],
            ]),
            )
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          if (this.isInitialLoad) {
            this.throbber.stop();
            setLoadingState(false, this.store);
            this.isInitialLoad = false;
          }
          this.loading = false;
          if (!this.parseData(data as InferenceProfileTable[])) return;
          this.updateView();
        });
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
