import {Component, Input, OnDestroy} from '@angular/core';
import {Store} from '@ngrx/store';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {getActiveHeapObjectState} from 'org_xprof/frontend/app/store/selectors';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A buffer details view component. */
@Component({
  standalone: false,
  selector: 'buffer-details',
  templateUrl: './buffer_details.ng.html',
  styleUrls: ['./buffer_details.scss']
})
export class BufferDetails implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  /** Selected module name in memory viewer */
  @Input() selectedModule = '';
  /** The session id */
  @Input() sessionId = '';

  heapObject: HeapObject|null = null;
  instructionName?: string;
  opcode?: string;
  size?: string;
  unpaddedSize?: string;
  padding?: string;
  expansion?: string;
  shape?: string;
  tfOpName?: string;
  groupName?: string;
  color?: string;
  sourceInfo?: string;

  constructor(private readonly store: Store<{}>) {
    this.store.select(getActiveHeapObjectState)
        .pipe(takeUntil(this.destroyed))
        .subscribe((heapObject: HeapObject|null) => {
          this.update(heapObject);
        });
  }

  hasValidGraphViewerLink() {
    return this.instructionName && this.selectedModule && this.sessionId;
  }

  getGraphViewerLink() {
    return `/graph_viewer/${this.sessionId}?module_name=${
        this.selectedModule}&node_name=${this.instructionName}`;
  }

  update(heapObject: HeapObject|null) {
    this.heapObject = heapObject;
    if (!heapObject) {
      return;
    }
    const sizeMiB = heapObject.sizeMiB || 0;
    const unpaddedSizeMiB = heapObject.unpaddedSizeMiB || 0;
    this.instructionName = heapObject.instructionName;
    this.opcode = heapObject.opcode ? heapObject.opcode + ' operation' : '';
    this.size = sizeMiB.toFixed(2);
    this.shape = heapObject.shape;
    this.tfOpName = heapObject.tfOpName;
    this.groupName = heapObject.groupName;
    this.sourceInfo = heapObject.sourceInfo;
    if (unpaddedSizeMiB) {
      this.unpaddedSize = unpaddedSizeMiB.toFixed(2);
      const utilization = unpaddedSizeMiB / sizeMiB;
      this.color = utils.flameColor(utilization, 0.7);
      if (utilization < 1) {
        this.expansion = (1 / utilization).toFixed(1);
        this.padding = (sizeMiB - unpaddedSizeMiB).toFixed(2);
      } else {
        this.expansion = '';
        this.padding = '';
      }
    } else {
      this.unpaddedSize = '';
      this.padding = '';
      this.expansion = '';
      this.color = 'rgb(192,192,192)';
    }
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
