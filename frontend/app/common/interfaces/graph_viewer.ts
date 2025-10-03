/** The interface for user input when generating a hlo graph */
export declare interface GraphConfigInput {
  selectedModule: string;
  opName: string;
  graphWidth: number;
  showMetadata: boolean;
  mergeFusion: boolean;
  useOriginalHloProto: boolean;
  programId?: string;
  graphType?: string;
  symbolId?: string;
  symbolType?: string;
}

/** The query parameter object for route navigation and xhr */
export declare interface GraphViewerQueryParams {
  node_name: string;
  module_name: string;
  graph_width: number;
  show_metadata: boolean;
  merge_fusion: boolean;
  use_original_hlo_proto: boolean;
  program_id?: string;
  graph_type?: string;
  symbol_id?: string;
  symbol_type?: string;
  show_me_graph?: boolean;
}

/** The interface for graph type object for selection */
export declare interface GraphTypeObject {
  value: string;
  label: string;
}
