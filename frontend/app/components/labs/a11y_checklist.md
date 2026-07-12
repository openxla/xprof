# XProf Labs accessibility (a11y) checklist

**Scope:** Material 3 Labs marketplace + playground (`frontend/app/components/labs/`)
and curated lab tools under `labs/` (e.g. Memory Analysis).

**Source:** FIX-041 / critique of Labs UI landings (#2892 and follow-ups).

**Status:** Partial pass — critical icon-control labels addressed; remaining items
tracked as debt below. Labs remains behind the experimental `labs` feature flag;
do not expand surface area without updating this checklist.

## How to re-run a smoke pass

1. Enable Labs in the profiler UI (experimental flag / tool entry that loads
   `xprof-labs`).
2. Keyboard: Tab through header → tabs → filter chips → search → view toggle →
   cards → Launch / more / favorite.
3. Screen reader (optional): VoiceOver / NVDA on the same path; confirm name,
   role, and state for toggles and icon-only buttons.
4. Curated tool: open Memory Analysis and Tab through sidebar toggle, hierarchy /
   metric / color toggles, tabs, search, table, zoom controls.

## Checklist

| Area | Criterion | Status | Notes |
|------|-----------|--------|-------|
| Feature flag | Labs stays experimental; no a11y-only unblock of flag | **OK** | `isLabsEnabled()` gate unchanged |
| Page structure | One `h1` ("XProf Labs"); sections use headings | **OK** | Header `h1` present |
| Search field | Accessible name on marketplace search | **OK** | `aria-label="Search experiments"` |
| Playground code | Textarea named | **OK** | `aria-label="Python trace visualization code"` |
| Playground AI prompt | Input named | **OK** | `aria-label="AI prompt for generating visualization code"` |
| Filter chips | Clear name + pressed state | **Fixed** | Text label + `aria-pressed` |
| Icon-only controls | Every icon button has `aria-label` (not `title` alone) | **Fixed** | Discard, clear search, grid/list, favorite, more options |
| Text buttons | Visible text is sufficient name | **OK** | Back, Start experiment, Launch, Run, Ask AI, tabs |
| Keyboard — tabs | Curated / My Experiments operable with keyboard | **Partial** | Native `<button>` receives focus; not `role="tablist"` yet |
| Keyboard — chips | Category filters focusable and activatable | **OK** | Native `<button>`; Enter/Space activate |
| Keyboard — cards | Launch reachable; favorite / more do not trap focus | **OK** | Buttons in tab order |
| Focus visible | `:focus-visible` / Material focus rings usable | **Debt** | Rely on browser/Material defaults; custom SCSS may dull rings |
| Live regions | Search result count / AI generating announced | **Debt** | Result header is visual only; spinner not `aria-live` |
| Color contrast | Body text and chip active states meet WCAG AA | **Debt** | Not audited in this pass; Material 3 tokens assumed |
| SVG visuals (Memory Analysis) | Flame/treemap not sole channel for data | **Debt** | Table provides parallel data; charts are mouse-heavy |
| Memory Analysis toggles | Hierarchy / size / color groups labeled | **OK** | Existing `aria-label` on `mat-button-toggle-group` |
| Memory Analysis icon buttons | Sidebar + zoom controls labeled | **Fixed** | `aria-label` on icon buttons |
| Loading indicators | Progress / spinner named | **Partial** | Memory Analysis spinner labeled; marketplace has no top-level bar |
| Disabled actions | Disabled review/draft convey why when focused | **Debt** | `[disabled]="true"` only; no `aria-describedby` |

## Critical gaps fixed in this pass

- Icon-only Labs marketplace / playground controls: `aria-label` added (previously
  `title` only, which is weak for AT).
- Filter chips: `aria-pressed` bound to selection so toggle state is exposed.
- Memory Analysis: sidebar toggle, flame-graph and treemap zoom/reset icon
  buttons, and loading spinner named for AT.

## Remaining debt (file or follow up)

1. **Tab pattern:** Promote `.labs-tabs` to a proper tabs widget
   (`role="tablist"`, `role="tab"`, `aria-selected`, `role="tabpanel"`, arrow-key
   navigation) instead of two independent buttons.
2. **Focus management:** Moving experiments ↔ playground should move focus to the
   new view heading (or first control) so keyboard users are not left on a
   removed control.
3. **Playground chart:** Decorative bar mock should be `aria-hidden` or replaced
   with a real chart that exposes data via text/table.
4. **Contrast audit:** Run axe / Lighthouse on Labs + Memory Analysis with light
   (and dark, if applicable) themes; fix chip and muted text ratios.
5. **SVG interaction:** Flame graph / treemap nodes are clickable `<g>` without
   keyboard equivalents; consider focusable nodes or rely on the buffers table
   as the accessible path and document that contract.
6. **Favorites semantics:** Star control is a toggle; long-term use
   `aria-pressed` (already partially modeled) and persist announcement of state
   change via snackbar `polite` live region if needed.
7. **"More options" menu:** Control is non-functional; when a menu is wired, use
   `mat-menu` with `aria-haspopup` / `aria-expanded`.

## Out of scope

- Full Material 3 redesign of Labs.
- Removing or default-enabling the Labs experimental flag.
- Automated a11y CI (axe in browser tests) — useful follow-up, not required here.
