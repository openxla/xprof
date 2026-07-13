/**
 * Enum representing the hardware components.
 */
export enum ComponentType {
  TC = 'tc',
  SCS = 'scs',
  SCTC = 'sctc',
  SCTD = 'sctd',
  CMN = 'cmn',
  ICR = 'icr',
}

/**
 * Map from ComponentType to its corresponding group name in the form.
 */
export const COMPONENT_TO_GROUP_NAME: Record<ComponentType, string> = {
  [ComponentType.TC]: 'tc_sampling',
  [ComponentType.SCS]: 'scs_sampling',
  [ComponentType.SCTC]: 'sctc_sampling',
  [ComponentType.SCTD]: 'sctd_sampling',
  [ComponentType.CMN]: 'cmn_sampling',
  [ComponentType.ICR]: 'icr_sampling',
};

/**
 * Map from ComponentType to its corresponding search keyword for counters.
 * Currently, these keywords apply to TPU v7x only as no others are supported.
 */
export const COMPONENT_TO_SEARCH_KEYWORD: Record<ComponentType, string> = {
  [ComponentType.TC]:
    'vf_chip_die0_tc_tcs_tc_misc_tcs_stats_tcs_stats_counters_unprivileged',
  [ComponentType.SCS]: 'vf_chip_die0_sc_0_scs_sc_stats_counters',
  [ComponentType.SCTC]: 'vf_chip_die0_sc_0_sctc_0_sc_stats_counters',
  [ComponentType.SCTD]: 'vf_chip_die0_sc_0_sctd_0_sc_stats_counters',
  [ComponentType.CMN]:
    'vf_chip_die0_cmn_cmnur_0_cmn_stats_debug_fixed_stats_counters',
  [ComponentType.ICR]:
    'vf_chip_chiplet_icr_icr_data_0_debug_domain_icr_data_stats_packet_counters',
};
