# Transcript formatting rules

These rules are loaded into the transcript post-processing agent's instructions.

Note: the rules below are written for German transcripts. The post-processing
agent is instructed to apply them only when the transcript is German.
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

## Phone numbers
- Write Swiss mobile and phone numbers grouped as `079 123 45 67` (3-3-2-2 digit groups separated by spaces).
- Never write the digits comma-separated or as prose (`079, 123, 45, 67` or `null sieben nein` stay wrong): join the spoken digits into one number and group them `079 123 45 67`.

## E-mail addresses
- Write spoken e-mail addresses as a single proper address: `blabla@blabla.ch`.
- Replace spoken forms of the symbols with the symbols themselves (`at`/`ät` -> `@`, `punkt`/`dot` -> `.`) and remove the spaces between the parts (`blabla at blabla punkt ch` -> `blabla@blabla.ch`), all lowercase.

## Serial numbers and identifier codes
- Write spoken sequences of digits and letters that identify something — serial numbers, device numbers, IMEI, MAC addresses, article or reference codes — as ONE code with `-` between the spoken groups: `12B, 34, 17 18` -> `12B-34-17-18`.
- Never write such sequences comma-separated or as loose digit groups.
- Phone numbers are NOT serial numbers: they follow the phone number rule above. Dates, times, and currency amounts also keep their own rules.
