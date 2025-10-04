import {Component} from '@angular/core';
import {MatDialogRef} from '@angular/material/dialog';

/** A capture profile dialog component. */
@Component({
  standalone: false,
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
  traceMode = 'TRACE_COMPUTE_AND_SYNC';
  hostTracerTooltip = 'lower trace level to reduce amount of host traces ' +
      'collected, some tools will not function well when the host tracer ' +
      'level is less than info';
  deviceTracerLevel = '1';
  pythonTracerLevel = '0';
  delay = 0;

  constructor(private readonly dialogRef:
                  MatDialogRef<CaptureProfileDialog>) {}

  addressTypeChanged(value: string) {
    this.isTpuName = value === 'tpu';
  }

  serviceAddrChanged(value: string) {
    this.serviceAddr = value.trim();
  }

  captureProfile() {
    this.dialogRef.close({
      serviceAddr: this.serviceAddr,
      isTpuName: this.isTpuName,
      duration: this.duration,
      numRetry: this.numRetry,
      workerList: this.workerList,
      hostTracerLevel: Number(this.hostTracerLevel),
      deviceTracerLevel: Number(this.deviceTracerLevel),
      pythonTracerLevel: Number(this.pythonTracerLevel),
      delay: this.delay,
    });
  }

  close() {
    this.dialogRef.close();
  }
}
