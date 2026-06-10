# Contributing to RECAL

Thank you for your interest.  Contributions are welcome via pull request.

---

## Running the tests

```bash
# Install package and dev dependencies
pip install -e ".[dev]"

# Run fast tests (no real data required)
pytest -v -m "not slow"

# Run all tests including slow integration tests
pytest -v
```

Tests live in `tests/` (regression tests) and `recal_core/tests/` (unit tests).

### Linting

```bash
ruff check .
```

The CI workflow runs both `pytest -v -m "not slow"` and `ruff check .` on
Python 3.10, 3.11, and 3.12.  A PR must pass all three before merging.

---

## Adding a new model backend

1. Read [docs/MODEL_FORMAT.md](docs/MODEL_FORMAT.md) for the expected contract
   (`.predict_proba(X) -> np.ndarray`).
2. Add a loader branch in `recal_cli/model_loader.py` — follow the pattern of
   existing backends.
3. Add at least one test in `tests/` that loads a minimal model of that type
   and calls `predict_proba` on synthetic data.
4. Update the backend table in [docs/MODEL_FORMAT.md](docs/MODEL_FORMAT.md).

If the new backend requires an optional dependency, add it under
`[project.optional-dependencies]` in `pyproject.toml` (e.g., `torch = ["torch>=2.0"]`).

---

## Submitting a pull request

1. Fork the repository and create a branch from `main`.
2. Make your changes.  Keep commits focused; one logical change per commit.
3. Run `ruff check .` and `pytest -v -m "not slow"` locally.
4. Open a PR against `main` with a clear description of what changed and why.
5. Reference any relevant issue numbers.

---

## Breaking change policy

RECAL follows [Semantic Versioning](https://semver.org).

- **Patch** (`0.2.x`) — bug fixes, documentation updates.
- **Minor** (`0.x.0`) — new features that are backwards-compatible.
- **Major** (`x.0.0`) — breaking changes to the public API (config schema,
  CLI flags, wrapper `.predict_proba` signature, serialisation format).

Breaking changes to the YAML config schema or the `AdaptedModelWrapper`
serialisation format require a major version bump and a migration note in
`CHANGELOG.md`.

---

*For project architecture see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).*
