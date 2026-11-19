import {Component, ChangeDetectionStrategy} from '@angular/core';
import {MatDialogRef} from '@angular/material/dialog';
import {parseFiniteNumber} from 'org_xprof/frontend/app/common/utils/utils';

/** A capture profile dialog component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'capture-profile-dialog',
  templateUrl: './capture_profile_dialog.ng.html',
  styleUrls: ['./capture_profile_dialog.scss']
})
export class CaptureProfileDialog {
  captureButtonLabel = 'Capture';
  closeButtonLabel = 'Close';
  serviceAddr = '';
  isTpuName = false;
  addressType = 'ip';
  duration = 1000;
  numRetry = 3;
  workerList = '';
  hostTracerLevel = '2';
  hostTracerTooltip = 'lower trace level to reduce amount of host traces ' +
      'collected, some tools will not function well when the host tracer ' +
      'level is less than info';
  deviceTracerLevel = '1';
  pythonTracerLevel = '0';
  delay = 0;
  extraOptions: Array<{key: string, value: string}> = [];

  constructor(private readonly dialogRef:
                  MatDialogRef<CaptureProfileDialog>) {}

  addressTypeChanged(value: string) {
    this.isTpuName = value === 'tpu';
  }

  serviceAddrChanged(value: string) {
    this.serviceAddr = value.trim();
  }

  captureProfile() {
    const options: {[key: string]: string|number|boolean} = {
      'serviceAddr': this.serviceAddr,
      'isTpuName': this.isTpuName,
      'duration': parseFiniteNumber(this.duration, 1000),
      'numRetry': parseFiniteNumber(this.numRetry, 3),
      'workerList': this.workerList,
      'hostTracerLevel': parseFiniteNumber(this.hostTracerLevel, 2),
      'deviceTracerLevel': parseFiniteNumber(this.deviceTracerLevel, 1),
      'pythonTracerLevel': parseFiniteNumber(this.pythonTracerLevel, 0),
      'delay': parseFiniteNumber(this.delay, 0),
    };

    for (const option of this.extraOptions) {
      const key = option.key.trim();
      if (key && !Object.prototype.hasOwnProperty.call(options, key)) {
        options[key] = option.value;
      }
    }

    this.dialogRef.close(options);
  }

  close() {
    this.dialogRef.close();
  }

  addExtraOption() {
    this.extraOptions.push({key: '', value: ''});
  }

  removeExtraOption(index: number) {
    this.extraOptions.splice(index, 1);
  }
}
