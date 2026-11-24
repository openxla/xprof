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
  selectedStepIndex = 0;
  maxStepIndex = 0;
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

  protected clearActivePodViewerInfo() {
    this.store.dispatch(
      setActivePodViewerInfoAction({activePodViewerInfo: null}),
    );
  }

  private resetStepData() {
    this.selectedStep = '';
    this.allReduceOpDb = undefined;
    this.allReduceOpChartData = undefined;
    this.channelDb = undefined;
    this.channelDbForChart = undefined;
    this.channelChartData = undefined;
    this.coreIdToReplicaIdMap = undefined;
    this.podStatsPerCore = undefined;
    this.podStatsForChart = undefined;
    this.podStatsChartData = undefined;
  }

  private updateSteps() {
    this.selectedStepIndex = 0;
    this.maxStepIndex = 0;
    const podStatsMap = this.data?.podStatsSequence?.podStatsMap;
    if (!podStatsMap || podStatsMap.length === 0) {
      this.updateSelectedStep(0);
      return;
    }
    this.maxStepIndex = podStatsMap.length - 1;
    this.updateSelectedStep(0);
  }

  updateSelectedStep(stepIndex: number) {
    this.clearActivePodViewerInfo();
    const podStatsMap = this.data?.podStatsSequence?.podStatsMap;
    if (
      !podStatsMap ||
      !Number.isInteger(stepIndex) ||
      stepIndex < 0 ||
      stepIndex >= podStatsMap.length
    ) {
      this.selectedStepIndex = 0;
      this.resetStepData();
      return;
    }

    const podStats = podStatsMap[stepIndex];
    if (!podStats) {
      this.selectedStepIndex = 0;
      this.resetStepData();
      return;
    }
    this.selectedStepIndex = stepIndex;
    const stepNum = podStats.stepNum ?? 0;
    // Negative step number indicates incomplete step.
    this.selectedStep = stepNum >= 0 ? stepNum.toString() : 'incomplete step';
    this.coreIdToReplicaIdMap = podStats.coreIdToReplicaIdMap || {};
    this.processAllReduceOpChart(podStats);
    this.processChannelDb(podStats);
    this.podStatsPerCore = podStats.podStatsPerCore;
    if (this.podStatsPerCore) {
      this.processStepBreakdownChart(this.podStatsPerCore);
    } else {
      this.podStatsForChart = undefined;
      this.podStatsChartData = undefined;
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
    this.channelDb = (podStats.channelDb || [])
      .slice(0)
      .sort((a, b) => ((a.channelId || '0') > (b.channelId || '0') ? 1 : -1));
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
        activePodViewerInfo: this.allReduceOpDb?.[allReduceOpIndex] ?? null,
      }),
    );
  }

  selectedChannelDb(channelDbIndex: number) {
    this.store.dispatch(
      setActivePodViewerInfoAction({
        activePodViewerInfo: this.channelDb?.[channelDbIndex] ?? null,
      }),
    );
  }

  selectedChannelChart(channelIndex: number) {
    this.store.dispatch(
      setActivePodViewerInfoAction({
        activePodViewerInfo: this.channelDbForChart?.[channelIndex] ?? null,
      }),
    );
  }

  convertRecordForDetailsCard(record: PodStatsRecord): PodStatsRecord {
    // Set an invalid chip id on non-TPU so the details card only shows hostname.
    const chipId = this.isTPU ? record.chipId : -1;
    if (!record.stepBreakdownUs) {
      return {...record, chipId};
    }
    const breakdown: {[key: /* uint32 */ string]: /* double */ number} = {};
    for (const event of this.stepBreakdownEvents) {
      breakdown[event.name] = record.stepBreakdownUs[event.id] || 0;
    }
    return {...record, chipId, stepBreakdownUs: breakdown};
  }

  selectedPodStatsChart(podStatsIndex: number) {
    const record = this.podStatsForChart?.[podStatsIndex];
    this.store.dispatch(
      setActivePodViewerInfoAction({
        activePodViewerInfo: record
          ? this.convertRecordForDetailsCard(record)
          : null,
      }),
    );
  }

  setDiagnostics(data: PodViewerDatabase | null) {
    this.diagnostics = {
      info: data?.diagnostics?.info || [],
      warnings: data?.diagnostics?.warnings || [],
      errors: data?.diagnostics?.errors || [],
    };
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
