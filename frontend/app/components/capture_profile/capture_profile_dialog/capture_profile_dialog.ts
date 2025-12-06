import {Component} from '@angular/core';
import {MatDialogRef} from '@angular/material/dialog';
import {ProfileOptions} from 'google3/third_party/tensorflow/tsl/profiler/protobuf/profiler_options.proto';

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
  addressType = 'ip';
  deviceType = 'gpu';
  duration = 1000;
  numRetry = 3;
  workerList = '';
  hostTracerLevel = '2';
  readonly TraceMode = ProfileOptions.TraceMode;
  traceMode: ProfileOptions.TraceMode = this.TraceMode.TRACE_COMPUTE;
  hostTracerTooltip = 'lower trace level to reduce amount of host traces ' +
      'collected, some tools will not function well when the host tracer ' +
      'level is less than info';
  deviceTracerLevel = '1';
  pythonTracerLevel = '0';
  delay = 0;
  extraOptions: Array<{key: string, value: string}> = [];

  constructor(private readonly dialogRef:
                  MatDialogRef<CaptureProfileDialog>) {}

  serviceAddrChanged(value: string) {
    this.serviceAddr = value.trim();
  }

  captureProfile() {
    const options: {[key: string]: string|number|boolean} = {
      serviceAddr: this.serviceAddr,

      duration: this.duration,
      numRetry: this.numRetry,
      workerList: this.workerList,
      hostTracerLevel: Number(this.hostTracerLevel),
      deviceTracerLevel: Number(this.deviceTracerLevel),
      pythonTracerLevel: Number(this.pythonTracerLevel),
      delay: this.delay,
    };
    // TraceMode is applicable when deviceType is 'tpu', regardless of whether
    // serviceAddr is a TPU name or IP addresses.
    if (this.deviceType === 'tpu') {
      options['traceMode'] = this.traceMode;
    }

    for (const option of this.extraOptions) {
      options[option.key] = option.value;
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
