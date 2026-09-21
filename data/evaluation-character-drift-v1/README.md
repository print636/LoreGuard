# Character Drift v1 DEV fixture

This directory contains an original, synthetic, developer-visible Simplified
Chinese fixture for the first character-consistency contract. It is not human
blind annotation and must not be reported as production accuracy.

- `dev.jsonl` is the only implemented split. Its expected labels may be used
  for contract development.
- No HOLDOUT answers exist in this stage. A future holdout must be authored,
  frozen and hashed before its first evaluation run; implementation debugging
  must never load its expected answers.
- `scripts/freeze_character_drift_eval.py` reports content hashes without
  opening an absent split or modifying fixture files.

The fixture tests semantic boundaries rather than prose quality: explicit
preference opposition, repeated versus one-off behavior, growth events,
disguise, temporary state, context, branch uncertainty and incomplete input.
