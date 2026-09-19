# Revision review flow

This page-specific guidance extends `../MASTER.md` for the revision and recheck loop.

## Compact design plan

- **Color:** inherit page deep `#090716`, surface `#1D1739`, lavender `#B7A2FF`,
  cyan `#8BE9F0`, warning `#FFBD82` and success `#8DE5B8`. Outcome text and
  labels always accompany color.
- **Type:** Noto Sans SC for controls and evidence; Noto Serif SC only for the
  page and stage titles; Cascadia Code only for run identifiers and coordinates.
- **Layout:** one vertical editorial ledger: baseline context, four-step strip,
  the current task, then a flat evidence comparison. Avoid a dashboard or chat.
- **Principles:** preserve the baseline run identity; make document changes
  inspectable before spending model tokens; describe machine comparison as
  detection status, not truth; keep upload and retry failures recoverable.

Desktop wireframe:

```text
┌ Baseline title + return ─────────────────────────────┐
├ 1 Review │ 2 Upload │ 3 Recheck │ 4 Compare          │
├ Project │ Baseline run │ Frozen docs │ Baseline issues│
├ Current stage: one continuous editorial work surface ┤
│ selected issue / upload form / input delta / results │
│ before evidence                   after evidence      │
└───────────────────────────────────────────────────────┘
```

Mobile stacks the context and evidence columns; the step strip becomes a 2×2
grid. It must not introduce page-level horizontal scrolling.

## Deliberate restraint

The memorable prismatic treatment remains reserved for `开始校验` throughout the
product. `开始复检` uses a quiet solid lavender action; steps, outcome filters and
upload actions stay flat. The flow uses no book spirit, ambient particles or card
grid, because evidence comparison needs a quiet reading surface.

## Meaning and copy

- `本次未再检出` means only that no match appeared in the target report. Never
  render it as `已解决` or merge it with a user's historical `已解决` feedback.
- `无法确认` covers degraded, ambiguous or otherwise non-comparable records.
- Show comparability status and reason beside totals. A degraded comparison never
  contributes to the no-longer-detected count.
- The exact route keeps the baseline run in the path and the selected issue,
  step, recheck run, outcome filter and visualization choices in whitelisted
  query state.
