// tslint:disable:enforce-comments-on-exported-symbols global enums
export enum IdleOption {
  YES = 'Yes',
  NO = 'No',
}

export enum OpExecutor {
  NONE = '',
  DEVICE = 'Device',
  HOST = 'Host'
}

export enum OpKind {
  NONE = '',
  NAME = 'Name',
  TYPE = 'Type'
}

export enum OpType {
  NONE = '',
  TENSORFLOW = 'TensorFlow',
  XLA_HLO = 'HLO',
}

export enum DataRequestType {
  UNKNOWN = 0,
  DATA_REQUEST_BEGIN,
  KERNEL_STATS,
  FRAMEWORK_OP_STATS,
  DATA_REQUEST_END,
  DIFF_DATA_REQUEST_BEGIN,
  FRAMEWORK_OP_STATS_DIFF,
  DIFF_DATA_REQUEST_END,
}

export enum FileExtensionType {
  UNKNOWN = '',
  PROTO_BINARY = 'pb',
  PROTO_TEXT = 'pbtxt',
  JSON = 'json',
  SHORT_TEXT = 'short_txt',
  LONG_TEXT = 'long_txt',
}
// tslint:enable:enforce-comments-on-exported-symbols
