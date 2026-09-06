# Contributing

## Workflow

1. Pull the latest `main` before starting:
   ```bash
   git checkout main
   git pull
   ```
2. Create a feature branch:
   ```bash
   git checkout -b feature/short-description
   ```
3. Commit with clear messages, push your branch, and open a Pull Request into `main`.
4. Request a review from a teammate before merging.
5. Delete the branch after merging.

## Branch naming

- `feature/...` — new functionality
- `fix/...` — bug fixes
- `docs/...` — documentation only
- `experiment/...` — exploratory notebooks/scripts

## Code style

- Keep functions small and readable; add docstrings for non-trivial logic.
- Run `flake8` locally before pushing if possible.
- Add tests under `tests/` for new logic where practical.

## Data & models

- Do not commit datasets or trained model weights — they're gitignored (`data/`, `models/`).
- Share large files via a shared drive/cloud storage and document how to fetch them in `docs/`.
