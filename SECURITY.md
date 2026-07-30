# Security Policy

## Supported versions

VisionDesk AI is pre-1.0. Only the latest release on the `main` branch receives
fixes.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Instead, use GitHub's private
["Report a vulnerability"](https://github.com/leohux/VisionDesk-AI/security/advisories/new)
flow. Include:

- affected version / commit
- a description of the issue and its impact
- steps to reproduce, if possible

You can expect an initial response within a few days.

## Scope notes

VisionDesk is **local-first**: screen capture, detection, and event memory stay
on your machine, and the core loop does not upload data. Be mindful that:

- Captured frames and the SQLite event store under `data/` may contain sensitive
  on-screen content. They are git-ignored by default — keep them that way.
- Enabling LocateAnything-3B loads third-party model weights; trust them as you
  would any external model.
- Bind the Gradio UI to localhost unless you intentionally add authentication
  and network controls.
- Review snapshots and logs before sharing them in an issue or pull request.
