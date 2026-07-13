/** Shared defaults for performance counter sampling per xprof_service.proto. */
export const DEFAULT_SAMPLING_INTERVAL_US = 10000;
/** Shared defaults for performance counter sampling per xprof_service.proto. */
export const DEFAULT_SAMPLING_IS_EXTERNAL_TRIGGER = false;
/** Shared defaults for performance counter sampling per xprof_service.proto. */
export const DEFAULT_SAMPLING_SCALING = 0;
/** Shared defaults for performance counter sampling per xprof_service.proto. */
export const DEFAULT_SAMPLING_COUNTER_SIZE_BITS = 3;

/** Mimics the PeriodicCounterSamplingOptions proto class from xprof_service.proto. */
// tslint:disable:enforce-name-casing
export declare interface PeriodicCounterSamplingOptions {
  interval_us?: number;
  is_external_trigger?: boolean;
  scaling?: number;
  counter_size_bits?: number;
  indices?: number[];
}
