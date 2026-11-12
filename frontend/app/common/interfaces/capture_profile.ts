/** The base interface for a response of capture profile api. */
export declare interface CaptureProfileResponse {
  error?: string;
  result?: string;
}

/** The base interface for a options of capture profile api. */
export declare interface CaptureProfileOptions {
  serviceAddr: string;
  isTpuName: boolean;
  duration: number;
  numRetry: number;
  workerList: string;
  hostTracerLevel: number;
  deviceTracerLevel: number;
  pythonTracerLevel: number;
  delay: number;
}

/** A serializable object with profiler configuration details. */
export declare interface ProfilerConfig {
  hideCaptureProfileButton: boolean;
  srcPathPrefix: string;
  enableTabNameLabel?: boolean;
}
