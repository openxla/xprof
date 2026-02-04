import {CommonModule} from '@angular/common';
import {Component, OnDestroy} from '@angular/core';
import {MatCardModule} from '@angular/material/card';
import {Store} from '@ngrx/store';

import {AllReduceOpInfo, ChannelInfo, PodStatsRecord} from 'org_xprof/frontend/app/common/interfaces/data_table';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {getActivePodViewerInfoState} from 'org_xprof/frontend/app/store/selectors';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

interface DetailInfo {
  title: string;
  value: string;
}

/** A pod viewer details view component. */
@Component({
  standalone: true,
  selector: 'pod-viewer-details',
  templateUrl: 'pod_viewer_details.ng.html',
  styleUrls: ['pod_viewer_details.scss'],
  imports: [CommonModule, MatCardModule],
})
export class PodViewerDetails implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  info?: AllReduceOpInfo|ChannelInfo|PodStatsRecord;
  name = '';
  details: DetailInfo[] = [];
  hloNames = '';
  replicaGroups = '';
  description = '';

  constructor(private readonly store: Store<{}>) {
    this.store.select(getActivePodViewerInfoState)
        .pipe(takeUntil(this.destroyed))
        .subscribe((info: AllReduceOpInfo|ChannelInfo|PodStatsRecord|null) => {
          this.update(info);
        });
  }

  private updateSizeAndLatency(info: AllReduceOpInfo|ChannelInfo) {
    const dataSize = Number(info.dataSize || '0');
    const latency = Number(info.durationUs || '0');
    this.details.push({
      title: 'Data Transferred',
      value: utils.bytesToMiB(dataSize).toFixed(2) + ' MiB',
    });
    this.details.push({
      title: 'Latency',
      value: latency.toFixed(2) + ' Us',
    });
    this.details.push({
      title: 'BW',
      value: (latency !== 0 ? dataSize / latency / 1073.74 : 0).toFixed(2) +
          ' GiB/s',
    });
  }

  private updateAllReduceOpInfo(info: AllReduceOpInfo) {
    this.info = info;
    this.name = info.name || '';
    this.updateSizeAndLatency(info);
    (info.replicaGroups || []).forEach(replicaGroup => {
      if (replicaGroup.replicaIds && replicaGroup.replicaIds.length > 0) {
        this.replicaGroups += '{' + replicaGroup.replicaIds.join(',') + '} ';
      }
    });
    this.description = info.description || '';
  }

  private updateChannelInfo(info: ChannelInfo) {
    this.info = info;
    this.name = 'Channel # ' + (info.channelId || 0).toString();
    this.updateSizeAndLatency(info);
    this.details.push({
      title: 'Send Delay',
      value:
          utils.bytesToMiB(Number(info.sendDelayUs || '0')).toFixed(2) + ' Us',
    });
    (info.hloNames || []).forEach(hloName => {
      if (hloName) {
        this.hloNames += '"' + hloName + '" ';
      }
    });
    this.description = info.description || '';
  }

  private updatePodStatsRecord(info: PodStatsRecord) {
    this.info = info;
    if (info.chipId === -1) {
      this.name = 'Step breakdown of ' + (info.hostName || 'localhost');
    } else {
      this.name = 'Step breakdown of chip ' + (info.chipId || 0).toString();
      if (info.hostName) {
        this.details.push({
          title: 'Hostname',
          value: info.hostName,
        });
      }
    }
    const total = info.totalDurationUs || 0;
    if (!total || !info.stepBreakdownUs) {
      return;
    }
    Object.keys(info.stepBreakdownUs).forEach((key) => {
      const value: number = utils.getPodStatsRecordBreakdownProperty(info, key);
      this.details.push({
        title: key,
        value: value.toFixed(2) + ' Us (' + (value / total * 100).toFixed(2) +
            '%)',
      });
    });
  }

  update(info: AllReduceOpInfo|ChannelInfo|PodStatsRecord|null) {
    this.details = [];
    this.hloNames = '';
    this.replicaGroups = '';
    if (!info) {
      this.info = undefined;
      return;
    }
    if (info.hasOwnProperty('channelId')) {
      this.updateChannelInfo(info as ChannelInfo);
    } else if (info.hasOwnProperty('chipId')) {
      this.updatePodStatsRecord(info as PodStatsRecord);
    } else {
      this.updateAllReduceOpInfo(info as AllReduceOpInfo);
    }
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
