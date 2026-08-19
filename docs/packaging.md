# Packaging and Releasing

LOFOP ships two ways for end users:

- **PyPI** — `pip install lofop`
- **AUR** (Arch Linux) — `yay -S lofop`

This document is for maintainers publishing releases. End users only need the install commands
above (see the [README](../README.md) / [MANUAL](../MANUAL.md)).

## Building the distributions

The package builds a standard wheel and sdist with no custom steps:

```bash
pip install build twine
python -m build              # writes dist/lofop-<ver>.tar.gz and .whl
python -m twine check dist/* # validates metadata
```

The sdist bundles the C++ ops source (`lofop/csrc/*.cpp`), the built-in configs, the license, and
the docs, so a source build is self-contained.

## Publishing to PyPI

Releases publish automatically via `.github/workflows/release.yml` when a version tag is pushed.
It uses **PyPI Trusted Publishing (OIDC)** — no API token is stored in the repo.

**One-time setup** (PyPI account required):

1. On https://pypi.org, register the `lofop` project name (or create a *pending publisher*).
2. Under the project's *Publishing* settings, add a trusted publisher:
   - Owner: `tedo001`, Repository: `LOFOP`, Workflow: `release.yml`, Environment: `pypi`.
3. In the GitHub repo, create an environment named `pypi` (Settings -> Environments).

**Each release:**

```bash
# 1. Bump the version in lofop/version.py AND pyproject.toml, commit, merge to main.
# 2. Tag and push:
git tag v1.2.1      # match the version in pyproject.toml
git push origin v1.2.1
```

The workflow builds, checks, and publishes. `pip install lofop` serves it within a minute.

To publish manually instead (needs a PyPI token):

```bash
python -m build
python -m twine upload dist/*
```

## Publishing to the AUR

The AUR package lives in [`packaging/aur/`](../packaging/aur/) and sources the PyPI sdist, so it
tracks the pip release. **Publish to PyPI first**, then:

```bash
cd packaging/aur
updpkgsums                                  # fills the real sha256 from the PyPI sdist
makepkg --printsrcinfo > .SRCINFO           # regenerate metadata
makepkg -si                                 # optional: build+install locally to test

# Push to the AUR (requires an AUR account + registered SSH key):
git clone ssh://aur@aur.archlinux.org/lofop.git aur-lofop
cp PKGBUILD .SRCINFO aur-lofop/
cd aur-lofop && git commit -am "lofop 1.2.1" && git push
```

After that, `yay -S lofop` (or any AUR helper) installs it. Optional features map to AUR
optdepends: `python-pytorch` for models/training, `python-onnx`/`python-onnxruntime` for export,
`gcc` for the native C++ ops.

## Version bumping

The version is defined in two places that must stay in sync: `lofop/version.py` (`__version__`)
and `pyproject.toml` (`version`). Update both, and the AUR `pkgver`, for each release.
