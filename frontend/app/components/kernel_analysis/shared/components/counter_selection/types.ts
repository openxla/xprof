/** Option for a single counter. */
export interface CounterOption {
  id: string;
  label: string;
  description?: string;
}

/** Group of counters, usually mapped to a functional unit. */
export interface CounterGroup {
  name: string;
  counters: CounterOption[];
  expandByDefault?: boolean;
}

/** Configuration for the counter selection component. */
export interface CounterSelectionConfig {
  groups: CounterGroup[];
  exactMatchForPureNumbers?: boolean;
}

/** Interface for raw counter data from backend. */
export declare interface Counter {
  name: string;
  val: number;
}
