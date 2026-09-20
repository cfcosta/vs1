# Steam purchase-history navigation

User-requested test on September 20, 2026, using the existing signed-in Chrome
session. “Statements” was interpreted as purchase history after an unanswered
clarification; wallet/market history was not tested. Jev used the upstream prompt
and the committed runtime from `e29d5c1d1097`, with a 30-action limit. No runtime
or prompt changes were made for these tests. No screenshots were requested.

| Starting point       | Outcome                      | Actions | Decisions | Loop time | Median decision |
| -------------------- | ---------------------------- | ------: | --------: | --------: | --------------: |
| Steam store homepage | BLOCKED; verification failed |       1 |         4 |    3.31 s |          305 ms |
| Account Details      | DONE; verification passed    |       1 |         3 |    2.32 s |          329 ms |

From the homepage, the agent clicked **View your profile**, landed on a community
profile, and declared BLOCKED. The initial decision request contained the account
menu button, labeled with the account name, alongside the profile link. A
read-only Chrome-CDP check confirmed the account menu is a visible native button.
The evidence points to incorrect target selection rather than omission of that
control. No menu interaction was manually injected into the agent run.

The second test explicitly started at Account Details to isolate the requested
history navigation. It clicked **View purchase history** and reached
`https://store.steampowered.com/account/history/`. Independent verification
required that host/path and a visible purchase-history heading; both passed.
This establishes navigation to history, not the completeness of transactions or
support for wallet/market statements. No transactions were opened, purchases
made, refunds requested, funds added, credentials entered, or settings changed.

Both runs used fresh agent-owned tabs, closed by the runtime afterward. The
existing user tab was only inspected. Raw traces contain private account/history
content and remain under ignored `artifacts/steam-account/`; the initial Chrome
snapshot remains under ignored `output/chrome-cdp/steam/`. Committed results
contain timings and the two generic action labels, without transaction details,
account identifiers, or profile URLs.

Reproduce with an already signed-in Chrome CDP session and `TYPESAFE_API_KEY`:

```sh
target/release/vs1-browser \
  --scenario research/steam-account/purchase-history.json \
  --prompt upstream --max-steps 30 --output artifacts/steam-home-retest

target/release/vs1-browser \
  --scenario research/steam-account/purchase-history-from-account.json \
  --prompt upstream --max-steps 30 --output artifacts/steam-account-retest
```

The scenarios instruct the agent to stop if login or Steam Guard is required.
Neither trial needed the text helper. These are single trials, not a reliability
estimate. See [results.json](results.json) for the reduced measurements.
