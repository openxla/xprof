import {Store} from '@ngrx/store';
import {
  AllReduceOpInfo,
  ChannelInfo,
  PodStatsMap,
  PodStatsRecord,
  PodViewerDatabase,
  PodViewerTopology,
  PrimitiveTypeNumberString,
  StepBreakdownEvent,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {setActivePodViewerInfoAction} from 'org_xprof/frontend/app/store/actions';

/** A common class of pod viewer component. */
export class PodViewerCommon {
  data: PodViewerDatabase | null = null;
  minStep = 0;
  maxStep = 0;
  selectedStep = '';
  allReduceOpDb?: AllReduceOpInfo[];
  allReduceOpChartData?: Array<Array<PrimitiveTypeNumberString | undefined>>;
  channelDb?: ChannelInfo[];
  channelDbForChart?: ChannelInfo[];
  channelChartData?: Array<Array<PrimitiveTypeNumberString | undefined>>;
  coreIdToReplicaIdMap?: {[key: /* uint32 */ string]: /* uint32 */ number};
  diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
  podStatsPerCore?: {[key: string]: PodStatsRecord};
  podStatsForChart?: PodStatsRecord[];
  podStatsChartData?: Array<Array<PrimitiveTypeNumberString | undefined>>;
  topology?: PodViewerTopology;
  stepBreakdownEvents: StepBreakdownEvent[] = [];
  stepBreakdownChartDescription = '';
  deviceType = '';
  isTPU = false;

  constructor(readonly store: Store<{}>) {}

  private updateSteps() {
    this.minStep = 0;
    this.maxStep = 0;
    if (
      !this.data ||
      !this.data.podStatsSequence ||
      !this.data.podStatsSequence.podStatsMap
    ) {
      this.updateSelectedStep(this.minStep);
      return;
    }
    const podStats = this.data.podStatsSequence.podStatsMap[0];
    if (!podStats) {
      this.updateSelectedStep(this.minStep);
      return;
    }
    this.minStep = podStats.stepNum || 0;
    this.maxStep = Math.max(
      this.minStep,
      this.minStep + this.data.podStatsSequence.podStatsMap.length - 1,
    );
    this.updateSelectedStep(this.minStep);
  }

  updateSelectedStep(step: number) {
    if (
      !this.data ||
      !this.data.podStatsSequence ||
      !this.data.podStatsSequence.podStatsMap
    ) {
      this.selectedStep = '';
      this.channelDb = undefined;
      this.podStatsPerCore = undefined;
      return;
    }

    const podStats =
      this.data.podStatsSequence.podStatsMap[step - this.minStep];
    if (!podStats) {
      this.selectedStep = '';
      this.channelDb = undefined;
      this.podStatsPerCore = undefined;
      return;
    }
    // Negative step number indicates incomplete step.
    this.selectedStep = step >= 0 ? step.toString() : 'incomplete step';
    this.coreIdToReplicaIdMap = podStats.coreIdToReplicaIdMap || {};
    this.processAllReduceOpChart(podStats);
    this.processChannelDb(podStats);
    this.podStatsPerCore = podStats.podStatsPerCore;
    if (this.podStatsPerCore) {
      this.processStepBreakdownChart(this.podStatsPerCore);
    }
  }

  processAllReduceOpChart(podStats: PodStatsMap) {
    this.allReduceOpDb = (podStats.allReduceOpDb || [])
      .slice(0)
      .sort((a, b) => ((a.durationUs || 0) < (b.durationUs || 0) ? 1 : -1));
    this.allReduceOpChartData = this.allReduceOpDb.map((allReduceOpInfo) => {
      let name = allReduceOpInfo.name || '';
      name = name.replace(/ll-reduce.|usion.|ll-reduce|usion/, '');
      name = name.replace(/all-to-all.|all-to-all/, 'l');
      name = name.length > 1 ? name : name + '0';
      return [name, allReduceOpInfo.durationUs];
    });
    this.allReduceOpChartData.unshift(['name', 'Duration (us)']);
  }

  processChannelDb(podStats: PodStatsMap) {
    this.channelDb = (podStats.channelDb || []).sort((a, b) =>
      (a.channelId || '0') > (b.channelId || '0') ? 1 : -1,
    );
    this.channelDbForChart = (podStats.channelDb || [])
      .slice(0)
      .sort((a, b) => ((a.durationUs || 0) < (b.durationUs || 0) ? 1 : -1));
    this.channelChartData = this.channelDbForChart.map((channelInfo) => [
      channelInfo.channelId,
      channelInfo.durationUs,
    ]);
    this.channelChartData.unshift(['channelId', 'Duration (us)']);
  }

  // Convert PodStatsRecord to gviz data row.
  parsePodStatsRecord(podStatsRecord: PodStatsRecord): Array<string | number> {
    const entry: Array<string | number> = [];
    if (this.isTPU) {
      entry.push(
        '(' +
          (podStatsRecord.chipId || 0).toString() +
          ',' +
          (podStatsRecord.nodeId || 0).toString() +
          ')',
      );
    } else {
      entry.push(podStatsRecord.hostName || 'localhost');
    }
    for (const event of this.stepBreakdownEvents || []) {
      entry.push(
        utils.getPodStatsRecordBreakdownProperty(
          podStatsRecord,
          event.id.toString(),
        ),
      );
    }
    return entry;
  }

  processStepBreakdownChart(podStatsPerCore: {
    [key: /* uint32 */ string]: PodStatsRecord;
  }) {
    this.podStatsForChart = Object.values(podStatsPerCore || {}).map(
      (podStatsRecord) => podStatsRecord,
    );
    if (this.isTPU) {
      this.podStatsForChart.sort((a, b) => {
        if (a.chipId === b.chipId) {
          return (a.nodeId || 0) > (b.nodeId || 0) ? 1 : -1;
        }
        return (a.chipId || 0) > (b.chipId || 0) ? 1 : -1;
      });
    } else {
      this.podStatsForChart.sort((a, b) => {
        return (a.hostName || '') > (b.hostName || '') ? 1 : -1;
      });
    }
    this.podStatsChartData = this.podStatsForChart.map(
      this.parsePodStatsRecord,
      this,
    );
    const metrics: string[] = this.stepBreakdownEvents.map(
      (event) => event.name,
    );
    metrics.unshift('metrics');
    this.podStatsChartData.unshift(metrics);
  }

  selectedAllReduceOpChart(allReduceOpIndex: number) {
    this.store.dispatch(
      setActivePodViewerInfoAction({
        activePodViewerInfo: this.allReduceOpDb
          ? this.allReduceOpDb[allReduceOpIndex]
          : null,
      }),
    );
  }

  selectedChannelDb(channelDbIndex: number) {
    this.store.dispatch(
      setActivePodViewerInfoAction({
        activePodViewerInfo: this.channelDb
          ? this.channelDb[channelDbIndex]
          : null,
      }),
    );
  }

  selectedChannelChart(channelIndex: number) {
    this.store.dispatch(
      setActivePodViewerInfoAction({
        activePodViewerInfo: this.channelDbForChart
          ? this.channelDbForChart[channelIndex]
          : null,
      }),
    );
  }

  convertRecordForDetailsCard(record: PodStatsRecord): PodStatsRecord {
    const breakdown: {[key: /* uint32 */ string]: /* double */ number} = {};
    if (!record.stepBreakdownUs) return record;
    if (!this.isTPU) {
      // Set an invalid chip id so that the details card only show hostname.
      record.chipId = -1;
    }
    for (const event of this.stepBreakdownEvents) {
      breakdown[event.name] = record.stepBreakdownUs[event.id] || 0;
    }
    return {...record, stepBreakdownUs: breakdown};
  }

  selectedPodStatsChart(podStatsIndex: number) {
    this.store.dispatch(
      setActivePodViewerInfoAction({
        activePodViewerInfo: this.podStatsForChart
          ? this.convertRecordForDetailsCard(
              this.podStatsForChart[podStatsIndex],
            )
          : null,
      }),
    );
  }

  setDiagnostics(data: PodViewerDatabase | null) {
    if (!data || !data.diagnostics) return;
    this.diagnostics.info = data.diagnostics.info || [];
    this.diagnostics.warnings = data.diagnostics.warnings || [];
    this.diagnostics.errors = data.diagnostics.errors || [];
  }

  setStepBreakdownChartDescription(isTPU: boolean) {
    if (isTPU) {
      this.stepBreakdownChartDescription =
        '(x-axis: global chip id, node id, y-axis: time(us))';
    } else {
      this.stepBreakdownChartDescription =
        '(x-axis: hostname, y-axis: time(us))';
    }
  }

  parseData(data: PodViewerDatabase | null) {
    this.data = data;
    this.deviceType = this.data ? this.data.deviceType || '' : '';
    this.isTPU = !this.deviceType.includes('GPU') && this.deviceType !== 'CPU';

    this.setDiagnostics(this.data);
    this.stepBreakdownEvents = this.data
      ? this.data.stepBreakdownEvents || []
      : [];
    this.updateSteps();
    this.setStepBreakdownChartDescription(this.isTPU);
    this.topology = this.data ? this.data.topology : undefined;
  }
}
