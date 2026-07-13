# Transcript formatting rules

These rules are loaded into the transcript-cleanup agent's instructions.

Note: the rules below are written for German transcripts. The cleanup agent
is instructed to apply them only when the transcript is German.
Edit this file to change how normalized entities are written; the agent only
applies a rule when the affected value is already present in the transcript —
rules never add information.

## Times
- Write times of day as `HH:MM Uhr` (e.g. `14:30 Uhr`, not `halb drei` -> keep spoken forms; only unify digit forms like `14.30`, `14:30h`, `1430 Uhr`).
- Write durations as spoken (`eine Stunde`, `30 Minuten`) — do not convert them.

## Currencies
- Write Swiss franc amounts as `<amount> Franken` with an apostrophe as thousands separator: `22'000 Franken` (not `22000 CHF`, `22.000 Fr.`, `Fr. 22000.-`).
- Other currencies analogous: `1'500 Euro`, `300 Dollar`.

## Numbers
- Keep numbers as transcribed unless the same value is written inconsistently across the transcript; then use the digit form with `'` thousands separator.

## Dates
- Write dates as `<day>. <month name> <year>` (e.g. `3. März 2026`) when a full date is spoken.
