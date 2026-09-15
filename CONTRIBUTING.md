# Contributing to LOFOP

Thanks for your interest in improving LOFOP. This guide covers how to set up a
development environment, the quality gates every change must pass, and how to
propose changes.

## Development setup

LOFOP has an optional native C++ ops library and optional PyTorch/ONNX
dependencies. For full development, install the test extras and build the
native ops:

```bash
python -m pip install -e ".[dev,models,deploy]"
python -c "from lofop.ops import build_native, backend; build_native(); print(backend())"
```

The core, data, and CLI layers are torch-free by design, so most contributions
can be developed and tested without a GPU.

## Quality gates

Every change must pass the same checks CI runs:

```bash
ruff check lofop tests benchmarks examples scripts   # lint
python -m pytest tests/ -q                            # tests
python -m pytest tests/ --cov --cov-report=term-missing  # tests + coverage
python scripts/check_version_sync.py                  # version consistency
```

Coverage is enforced in CI with a floor configured in `pyproject.toml`
(`[tool.coverage.report] fail_under`); keep new code covered.

- **Tests are required.** Every new module or behavior needs unit tests. Use
  the project test base (`from torch.testing._internal` is not used here; see
  existing tests for the plain `unittest`/`pytest` style). Device-generic
  numerics should be tested on CPU.
- **Lint must be clean.** Run `ruff check` (and `ruff check --fix` for
  autofixable issues) before committing.
- **Backward compatibility.** Public APIs (anything exported from a package
  `__init__`) must not break without a clear reason and a CHANGELOG entry. New
  dataclass fields should be defaulted.
- **Docs.** Update `MANUAL.md` and the relevant file under `docs/` when you add
  or change user-facing behavior.

## Coding style

- Type hints on public functions; concise, self-documenting code.
- Match the surrounding style and architectural patterns. Prefer reusing
  existing abstractions (registry, event bus, config) over new ones.
- ASCII only in new code comments.
- Keep the core/data/CLI layers importable without torch; put anything that
  needs PyTorch in the models/training/deploy layers.

## Proposing changes

1. Branch off the development branch and make focused, reviewable commits.
2. Add tests and docs alongside the code.
3. Add a `CHANGELOG.md` entry under "Unreleased".
4. Open a pull request describing the change, the motivation, and the exact
   test commands you ran (see the PR template).

## Reporting bugs and requesting features

Use the issue templates. A good bug report includes the LOFOP version
(`lofop version`), the output of `lofop doctor`, and a minimal reproduction.

By contributing, you agree that your contributions are licensed under the
project's Apache-2.0 license.

## Third-party code policy

LOFOP ships under Apache-2.0 and claims to be an original implementation.
Both depend on every contribution being either your own work or permissively
licensed with proper attribution. Contributions are checked against this
policy before review.

**Never acceptable.** Code copied or transcribed from a GPL-, AGPL-, or
otherwise copyleft-licensed project. These licences are incompatible with
Apache-2.0: accepting such code would force LOFOP itself to relicense.
Attribution does not cure this -- the code must be written independently.
This applies to reading another project's source and retyping it, not only to
literal copy-paste. Widely copied trackers and detector implementations are a
common source of this problem, so tracking and post-processing contributions
receive particular scrutiny.

**Acceptable with attribution.** Code under permissive licences (MIT,
BSD, Apache-2.0), provided you add the copyright notice, record the source in
`NOTICE`, and state what you changed.

**Always acceptable.** Your own implementation of a published algorithm or a
standard mathematical method, written from the mathematics or the paper.
Algorithms are not copyrightable; one author's particular expression of one
is. Write the method yourself rather than adapting a specific implementation,
and do not carry across another project's private attribute names or
hand-tuned magic constants -- those are fingerprints of copying, not of the
mathematics.

### Provenance statement

Every pull request must answer, in its description:

1. Is any part of this derived from another project? Which, and under what
   licence?
2. If you implemented a published method, which paper or reference did you
   work from?
3. Does this add a new runtime import? If so, which extra in
   `pyproject.toml` declares it?

"All original work, no external sources consulted" is a complete and
acceptable answer.

### Dependency rules

- The base install must stay at PyYAML and Pillow. Anything heavier belongs
  in an extra.
- Optional packages (`numpy`, `scipy`, `supervision`, `onnx`, `torch` outside
  its subsystems) must never be imported at module scope. Import them inside
  the function that needs them and raise `LofopError` naming the extra.
- A stock `pip install lofop` must never raise `ModuleNotFoundError`.

`tests/test_dependency_hygiene.py` enforces all of the above automatically,
including a scan for copyleft markers. A contribution that trips it will fail
CI.
