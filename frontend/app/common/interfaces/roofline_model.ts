import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** The base interfact for roofline model property. */
declare interface RooflineModelProperty {
  device_type?: string;
  megacore?: string;
  has_cmem?: string;
  has_merged_vmem?: string;
  peak_flop_rate?: string;
  peak_hbm_bw?: string;
  peak_cmem_read_bw?: string;
  peak_cmem_write_bw?: string;
  peak_vmem_read_bw?: string;
  peak_vmem_write_bw?: string;
  hbm_ridge_point?: string;
  cmem_read_ridge_point?: string;
  cmem_write_ridge_point?: string;
  vmem_read_ridge_point?: string;
  vmem_write_ridge_point?: string;
}

/** The base interfact for roofline model. */
export declare interface RooflineModelData extends SimpleDataTable {
  p: RooflineModelProperty;
}
