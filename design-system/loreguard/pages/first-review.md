# First review flow

This page-specific guidance extends `../MASTER.md` for the first validation loop.

- A project validation view has one prismatic action: `开始校验`. Project creation,
  import, retry, report navigation and filters remain quiet controls.
- Before launch, use one continuous review surface: scope explanation, flat active
  document rows, then model/degradation boundaries. Do not split these into a card
  dashboard.
- Every document row names the frozen version, document role and story scope.
- “Provider configured/connected” describes readiness only. “Model actually used”
  is a run result and appears only after diagnostics are available.
- Precise run URLs are part of the visual information architecture. A missing or
  cross-project run gets a recoverable 404 state and never silently shows the newest
  report.
- Report category, feedback status and selected issue are linkable query state.
  Filter labels remain visible; selection cannot rely on color alone.
- The no-project quick text experiment is a legacy entry surface and explicitly
  says that it creates an independent project. It never appears beside a project's
  primary validation action.
