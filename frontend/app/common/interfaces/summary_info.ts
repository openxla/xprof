/** The base interface for a information of performance summary view. */
export interface SummaryInfo {
  level?: number; // Top level metric is level 1, use to control styles
  title: string;
  descriptions?: string[];
  tooltip?: string;
  value?: string;
  valueColor?: string;
  propertyValues?: string[];
  // Nested child metrics info
  childrenInfo?: SummaryInfo[];
}

/** freeform k-v pair property interface */
export interface GeneralProps {
  [key: string]: string;
}

/**
 * Interface for performance summary configuration object.
 * Will be translated into SummaryInfo object.
 */
export interface SummaryInfoConfig {
  title: string;
  tooltip?: string;
  // goodMetric equals true means the higher the better
  goodMetric?: boolean;
  description?: string;
  valueKey?: string;
  sdvKey?: string;
  childrenInfoConfig?: SummaryInfoConfig[];
  // custom callback to read value
  // valueKey and getValue are mutual exclusive
  getValue?: (arg: google.visualization.DataObjectCell[]) => string;
  // custom callback function to get a nested metrics list
  getChildValues?: (
    arg: GeneralProps | google.visualization.DataObjectCell[],
  ) => string[];
  unit?: string;
  valueColor?: string;
  trainingOnly?: boolean;
  inferenceOnly?: boolean;
}
