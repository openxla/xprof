/** The base interface for a navigation event. */
export declare interface NavigationEvent {
  // TODO(jonahweaver): Replace run and tag, with sessionId and tool
  // respectively.
  run?: string;
  tag?: string;
  host?: string;
  hosts?: string[];
  // Graph Viewer crosslink params
  opName?: string;
  moduleName?: string;
  programId?: string;
  graphType?: string;
  // Memory viewer params
  memorySpaceColor?: string;
  // Navigation controlling params
  firstLoad?: boolean;
}
