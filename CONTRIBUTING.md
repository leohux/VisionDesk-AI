# Contributing to VisionDesk AI

Thanks for your interest in improving VisionDesk AI. Issues and pull requests
are welcome.

## Ground rules

- Keep changes focused. No drive-by refactors mixed into feature/bugfix PRs.
- If you change trigger logic or profile defaults, add a short note explaining why.
- For performance claims, attach `visiondesk.py replay … --json` output.
- Don't commit model weights, event databases, captured frames, or large media.
  These are already covered by `.gitignore`.

## Dev setup

```bash
git clone https://github.com/leohux/VisionDesk-AI.git
cd VisionDesk-AI

# Install CUDA PyTorch for your GPU first: https://pytorch.org
pip install -r requirements.txt
```

LocateAnything-3B is optional; YOLO-only mode runs without it. See the README
section "LocateAnything-3B setup" for pointing VisionDesk at a local checkout.

## Before opening a PR

```bash
# byte-compile (catches syntax errors)
python -m compileall -q .

# critical lint (same gate as CI)
ruff check --select E9,F63,F7,F82 .
```

CI runs these on Python 3.10 / 3.11 / 3.12. A full `ruff check .` is run too but
is currently non-blocking.

## Commit / PR style

- One logical change per commit; write imperative, descriptive subjects.
- Reference the issue you're fixing in the PR description.
- Describe how you verified the change (profile, video, before/after FPS or 3B
  call rate where relevant).

## Reporting bugs

Open an issue with your OS, GPU + VRAM, Python version, the profile used, and
the exact command or UI steps to reproduce.
