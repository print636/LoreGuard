# Model settings page override

This page follows `../MASTER.md`; only the rules below are page-specific.

- Route: `/app/settings/model`, parallel to account security and outside the
  project workspace shell.
- Use the account-settings ledger layout: quiet dividers, one readable column,
  and no decorative mascot or ambient animation.
- The saved API key is represented by a constant mask only. Never render a
  stored value, prefix or suffix; reveal applies only to the unsaved local input.
- A connection test is a bounded preflight, not proof that a full story review
  will succeed. Keep this boundary visible beside every test result.
- Save uses the solid lavender primary treatment. The prismatic gradient remains
  exclusive to **开始校验**.
- Destructive deletion is visually separated, requires confirmation, and returns
  focus to its trigger when cancelled.
- On narrow screens all metrics and actions stack without horizontal scrolling;
  touch controls remain at least 44px high.
