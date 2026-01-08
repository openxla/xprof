/** The query parameter object from user input for route navigation and xhr */
export declare interface GraphViewerQueryParams {
  node_name: string;
  module_name: string;
  graph_width: number;
  show_metadata: boolean;
  merge_fusion: boolean;
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
