/**
 * @fileoverview Constants for trace event argument keys.
 * These constants should be kept in sync with their C++ counterparts defined
 * in:
 * org_xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h
 */

/** Trace event argument key for HLO op. */
export const HLO_OP = 'hlo_op';
/** Trace event argument key for HLO module. */
export const HLO_MODULE = 'hlo_module';
/** Trace event argument key for HLO module id. */
export const HLO_MODULE_ID = 'hlo_module_id';
/** Trace event argument key for program id. */
export const PROGRAM_ID = 'program_id';
/** Trace event argument key for kernel details. */
export const KERNEL_DETAILS = 'kernel_details';
