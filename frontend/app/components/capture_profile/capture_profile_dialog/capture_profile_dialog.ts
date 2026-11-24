import {ChangeDetectionStrategy, Component} from '@angular/core';
import {MatDialogRef} from '@angular/material/dialog';
import type {MatInput} from '@angular/material/input';
import {parseFiniteNumber} from 'org_xprof/frontend/app/common/utils/utils';

const MAX_INT32 = 2147483647;
const MAX_NUM_RETRY = MAX_INT32 - 1;

function isValidOptionalIntegerInRange(
  value: unknown,
  min: number,
  max: number,
): boolean {
  if (value == null || (typeof value === 'string' && value.trim() === '')) {
    return true;
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return true;
  }
  return Number.isInteger(num) && num >= min && num <= max;
}

/** A capture profile dialog component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'capture-profile-dialog',
  templateUrl: './capture_profile_dialog.ng.html',
  styleUrls: ['./capture_profile_dialog.scss'],
})
export class CaptureProfileDialog {
  readonly maxInt32 = MAX_INT32;
  readonly maxNumRetry = MAX_NUM_RETRY;
  readonly durationErrorStateMatcher: MatInput['errorStateMatcher'] = {
    isErrorState: () => !this.isDurationValid(),
  };
  readonly numRetryErrorStateMatcher: MatInput['errorStateMatcher'] = {
    isErrorState: () => !this.isNumRetryValid(),
  };
  readonly delayErrorStateMatcher: MatInput['errorStateMatcher'] = {
    isErrorState: () => !this.isDelayValid(),
  };

  captureButtonLabel = 'Capture';
  closeButtonLabel = 'Close';
  serviceAddr = '';
  isTpuName = false;
  addressType = 'ip';
  duration = 1000;
  numRetry = 3;
  workerList = '';
  hostTracerLevel = '2';
  hostTracerTooltip =
    'lower trace level to reduce amount of host traces ' +
    'collected, some tools will not function well when the host tracer ' +
    'level is less than info';
  deviceTracerLevel = '1';
  pythonTracerLevel = '0';
  delay = 0;
  extraOptions: Array<{key: string; value: string}> = [];

  constructor(private readonly dialogRef: MatDialogRef<CaptureProfileDialog>) {}

  addressTypeChanged(value: string) {
    this.isTpuName = value === 'tpu';
  }

  serviceAddrChanged(value: string) {
    this.serviceAddr = value.trim();
  }

  isDurationValid(): boolean {
    return isValidOptionalIntegerInRange(this.duration, 1, MAX_INT32);
  }

  isNumRetryValid(): boolean {
    return isValidOptionalIntegerInRange(this.numRetry, 0, MAX_NUM_RETRY);
  }

  isDelayValid(): boolean {
    return isValidOptionalIntegerInRange(this.delay, 0, MAX_INT32);
  }

  areNumericFieldsValid(): boolean {
    return (
      this.isDurationValid() && this.isNumRetryValid() && this.isDelayValid()
    );
  }

  isFormValid(): boolean {
    return Boolean(this.serviceAddr.trim()) && this.areNumericFieldsValid();
  }

  captureProfile() {
    if (!this.areNumericFieldsValid()) {
      return;
    }
    const options: {[key: string]: string | number | boolean} = {
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
