/**
 * Identifies a link from a selected trace event to another XProf tool. Used as
 * a `trackBy` key and as a test id suffix.
 */
export type CrossToolLinkId = 'source_and_ir' | 'roofline' | 'hlo_graph';

/**
 * The part of a cross-tool link that does not depend on the selected event.
 */
export declare interface CrossToolLinkSpec {
  id: CrossToolLinkId;
  /** Tool id used to build the destination URL. */
  toolName: string;
  /** Short label shown on the chip, e.g. 'HLO graph'. */
  label: string;
  /** Material icon shown before the label. */
  icon: string;
}

/**
 * A link from a selected trace event to another XProf tool, ready to render.
 *
 * Exactly one of `href` and `disabledReason` is set. A link with a
 * `disabledReason` is still rendered, greyed out, so that users can tell the
 * difference between "this tool does not apply to this op" (the link is absent
 * altogether) and "this tool applies but is unavailable right now".
 */
export declare interface CrossToolLink extends CrossToolLinkSpec {
  /** Full sentence used for the tooltip and the aria-label. */
  description: string;
  /** Destination URL. Unset when the link cannot be followed. */
  href?: string;
  /** Why the link cannot be followed. Unset when `href` is set. */
  disabledReason?: string;
}
