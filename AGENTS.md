Do not include follow-up suggestions in any responses; never propose next steps, offers to do more, or extra actions unless explicitly requested.

Commit rules:

- Commit prefixes like `refactor` must be lowercase at all times.
- Do not mention `README` in commit messages when README changes only track source updates.
- Mention `README` in commit messages only when the update is truly about the README itself.
- Only commit/push when explicitly instructed to do so by the user.
- Do not mention benchmark CSV/HTML updates in commit messages.
- Each commit message must be a single line; no body or multiline messages.

Commit message scope and verification:

- Commit messages must describe only the diff since the current HEAD.
- Never summarize the whole session; scope commit text to the actual changes.
- Before proposing/creating a commit, inspect `git diff --name-status HEAD` (and optionally `git diff --stat HEAD`) to verify exactly what changed.
- When asked to report "what changed since last commit", use the HEAD-relative diff; do not mix in earlier, already-committed work.

Code comment rules:

- Write comments that describe the current state of the code and its intent.
- Avoid comments that narrate past changes, refactors, or TODO history (e.g., "switched to X", "no longer uses Y").
- Remove or update any comment that becomes stale after edits; comments must match the code as it exists now.
