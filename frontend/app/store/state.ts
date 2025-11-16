import {Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {DataRequestType} from 'org_xprof/frontend/app/common/constants/enums';
import {AllReduceOpInfo, ChannelInfo, PodStatsRecord} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import {HostMetadata} from 'org_xprof/frontend/app/common/interfaces/hosts';
import {RunToolsMap} from 'org_xprof/frontend/app/common/interfaces/tool';

/** Type for active heap object state */
type ActiveHeapObjectState = HeapObject|null;

/** State of memory viewer */
export interface MemoryViewerState {
  activeHeapObject: ActiveHeapObjectState;
}

/** Type for active op profile node state */
type ActiveOpProfileNodeState = Node|null;

/** State of op profile */
export interface OpProfileState {
  activeOpProfileNode: ActiveOpProfileNodeState;
  selectedOpNodeChain: string[];
  rootNode: ActiveOpProfileNodeState;
}

/** State of op analysis settings */
export interface OpAnalysisState {
  applyScalingFactor: boolean;
}

/** General State of the Profiling */
export interface ProfilingGeneralState {
  deviceType: string;
}

/** Type for active pod viewer info state */
type ActivePodViewerInfoState = AllReduceOpInfo|ChannelInfo|PodStatsRecord|null;

/** State of pod viewer */
export interface PodViewerState {
  activePodViewerInfo: ActivePodViewerInfoState;
}

/** Type for capturing profile state */
type CapturingProfileState = boolean;

/** State of loading */
export interface LoadingState {
  loading: boolean;
  message: string;
}

/** Type for current tool state */
type CurrentToolState = string;

/** Interface for tool info */
export interface ToolInfo {
  key: string;
  label: string;
  url?: string;
  version?: string;
  isRouteLink?: boolean;
  queryParams?: {[key: string]: string};
}

/** Type for tools info state */
export type ToolsInfoState = ToolInfo[]|[];

/** The interface of params of DataRequest */
interface DataRequestParams {
  run?: string;
  tag?: string;
  host?: string;
  tool?: string;
  sessionId?: string;
}

/** Type for export_as_csv state */
type ExportAsCsvState = string;

/** State of data request */
export interface DataRequest {
  type: DataRequestType;
  params: DataRequestParams;
  loadingMessage?: string;
}

/** Type for error message state */
type ErrorMessageState = string;

/** State object */
export interface AppState {
  memoryViewerState: MemoryViewerState;
  opProfileState: OpProfileState;
  podViewerState: PodViewerState;
  capturingProfile: CapturingProfileState;
  loadingState: LoadingState;
  toolsInfoState: ToolsInfoState;
  hostsState: HostMetadata[];
  tagsState: string[];
  currentTool: CurrentToolState;
  exportAsCsv: ExportAsCsvState;
  errorMessage: ErrorMessageState;
  // OSS only
  dataRequest: DataRequest;
  runToolsMap: RunToolsMap;
  currentRun: string;
  profilingGeneralState: ProfilingGeneralState;
  opAnalysisState: OpAnalysisState;
}

/** Initial state of active heap object */
const INIT_ACTIVE_HEAP_OBJECT_STATE: ActiveHeapObjectState = null;

/** Initial state object */
export const INIT_MEMORY_VIEWER_STATE: MemoryViewerState = {
  activeHeapObject: INIT_ACTIVE_HEAP_OBJECT_STATE,
};

/** Initial state of active op profile node */
const INIT_ACTIVE_OP_PROFILE_NODE_STATE: ActiveOpProfileNodeState = null;

/** Initial state of op profile */
export const INIT_OP_PROFILE_STATE: OpProfileState = {
  activeOpProfileNode: INIT_ACTIVE_OP_PROFILE_NODE_STATE,
  selectedOpNodeChain: [],
  rootNode: INIT_ACTIVE_OP_PROFILE_NODE_STATE,
};

/** Initial general profiling state */
export const INIT_PROFILING_GENERAL_STATE: ProfilingGeneralState = {
  deviceType: 'TPU',
};

/** Initial state of op analysis settings */
export const INIT_OP_ANALYSIS_STATE: OpAnalysisState = {
  applyScalingFactor: false,
};

/** Initial state of active pod viewer info */
const INIT_ACTIVE_POD_VIEWER_INFO_STATE: ActivePodViewerInfoState = null;

/** Initial state of pod viewer */
export const INIT_POD_VIEWER_STATE: PodViewerState = {
  activePodViewerInfo: INIT_ACTIVE_POD_VIEWER_INFO_STATE,
};

/** Initial state of capturing profile */
const INIT_CAPTURING_PROFILE_STATE: CapturingProfileState = false;

/** Initial state of loading */
export const INIT_LOADING_STATE: LoadingState = {
  loading: false,
  message: '',
};

/** Initial state of current tool */
const INIT_CURRENT_TOOL_STATE: CurrentToolState = '';

/** Initial state of tools info */
const INIT_TOOLS_INFO_STATE: ToolsInfoState = [];

/** Initial state of request data */
const INIT_REQUEST_DATA_STATE: DataRequest = {
  type: DataRequestType.UNKNOWN,
  params: {},
};

/** Initial state of export_as_csv */
const INIT_EXPORT_AS_CSV_STATE: ExportAsCsvState = '';

/** Initial state of error message */
const INIT_ERROR_MESSAGE_STATE: ErrorMessageState = '';

/** Initial state of run tools map */
const INIT_RUN_TOOLS_MAP: RunToolsMap = {};

/** Initial state of current run */
const INIT_CURRENT_RUN = '';

/** Initial state of hosts list */
const INIT_HOSTS_STATE: HostMetadata[] = [];

/** Initial state of tags list */
const INIT_TAGS_STATE: string[] = [];

/** Initial state object */
export const INIT_APP_STATE: AppState = {
  memoryViewerState: INIT_MEMORY_VIEWER_STATE,
  opProfileState: INIT_OP_PROFILE_STATE,
  podViewerState: INIT_POD_VIEWER_STATE,
  capturingProfile: INIT_CAPTURING_PROFILE_STATE,
  loadingState: INIT_LOADING_STATE,
  currentTool: INIT_CURRENT_TOOL_STATE,
  toolsInfoState: INIT_TOOLS_INFO_STATE,
  hostsState: INIT_HOSTS_STATE,
  tagsState: INIT_TAGS_STATE,
  dataRequest: INIT_REQUEST_DATA_STATE,
  exportAsCsv: INIT_EXPORT_AS_CSV_STATE,
  errorMessage: INIT_ERROR_MESSAGE_STATE,
  runToolsMap: INIT_RUN_TOOLS_MAP,
  currentRun: INIT_CURRENT_RUN,
  profilingGeneralState: INIT_PROFILING_GENERAL_STATE,
  opAnalysisState: INIT_OP_ANALYSIS_STATE,
};

/** Feature key for store */
export const STORE_KEY = 'root';
