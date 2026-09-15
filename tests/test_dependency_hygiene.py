"""Guards for LOFOP's packaging and licensing rules.

These tests encode promises the project makes to its users rather than
behaviour of any one function:

* a stock ``pip install lofop`` must never raise ``ModuleNotFoundError``;
* optional dependencies must be reachable only through declared extras, and
  must fail with an actionable :class:`LofopError` when absent;
* no copyleft-licensed source may enter an Apache-2.0 project.

They are deliberately written against the source text so they keep holding as
the code changes.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "lofop"
PROJECT_ROOT = PACKAGE_ROOT.parent

# Packages that must never be imported at module scope anywhere in lofop.
# torch is permitted only inside the subsystems that declare it.
OPTIONAL_PACKAGES = {"numpy", "scipy", "supervision", "onnx", "onnxruntime", "tensorrt"}

# Subsystems allowed to import torch at module scope; the rest of LOFOP --
# core, data, ops, mlops, tracking -- must stay importable without it.
# utils holds the model benchmark, which measures torch modules by definition.
TORCH_SUBSYSTEMS = {"models", "training", "deploy", "utils"}


def python_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def top_level_imports(path: Path) -> set[str]:
    """Modules imported at module scope (not inside a function or method)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` blocks never execute at runtime.
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    continue
    return names


class TestLazyOptionalDependencies:
    @pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.name))
    def test_no_optional_package_at_module_scope(self, path):
        offenders = top_level_imports(path) & OPTIONAL_PACKAGES
        assert not offenders, (
            f"{path.relative_to(PROJECT_ROOT)} imports {sorted(offenders)} at module "
            "scope; move it inside the function that needs it so a stock install "
            "keeps working"
        )

    def test_torch_confined_to_its_subsystems(self):
        for path in python_files():
            if "torch" not in top_level_imports(path):
                continue
            relative = path.relative_to(PACKAGE_ROOT)
            subsystem = relative.parts[0] if len(relative.parts) > 1 else relative.stem
            assert subsystem in TORCH_SUBSYSTEMS or relative.stem == "sdk", (
                f"{relative} imports torch at module scope but is outside the "
                f"torch subsystems {sorted(TORCH_SUBSYSTEMS)}"
            )

    def test_tracking_package_imports_without_numpy(self):
        # Run in a subprocess with numpy blocked, so the guarantee is checked
        # even on a developer machine that happens to have numpy installed.
        script = (
            "import sys\n"
            "class Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        return self if name.split('.')[0] == 'numpy' else None\n"
            "    def load_module(self, name):\n"
            "        raise ImportError('numpy blocked for this test')\n"
            "sys.meta_path.insert(0, Block())\n"
            "import lofop.tracking\n"
            "from lofop.tracking import associate\n"
            "assert associate([[0,0,10,10]], [[1,1,11,11]])[0] == [(0, 0)]\n"
            "from lofop.core.exceptions import LofopError\n"
            "try:\n"
            "    lofop.tracking.MotionEstimator()\n"
            "except LofopError as exc:\n"
            "    assert 'lofop[tracking]' in str(exc), str(exc)\n"
            "else:\n"
            "    raise AssertionError('expected a LofopError naming the extra')\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_association_needs_nothing_beyond_the_core(self):
        from lofop.tracking.association import associate

        pairs, _, _ = associate([[0, 0, 10, 10]], [[1, 1, 11, 11]])
        assert pairs == [(0, 0)]

    def test_assignment_needs_nothing_beyond_the_core(self):
        from lofop.ops.assignment import optimal_assignment

        assert optimal_assignment([[1.0, 5.0], [5.0, 1.0]]) == [0, 1]


class TestDeclaredExtras:
    def test_optional_packages_have_an_extra(self):
        text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for extra in ("tracking", "supervision", "models", "deploy"):
            assert f"{extra} = [" in text, f"pyproject.toml is missing the {extra!r} extra"

    def test_runtime_dependencies_stay_minimal(self):
        text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        runtime = text.split("dependencies = [", 1)[1].split("]", 1)[0]
        assert "numpy" not in runtime and "torch" not in runtime, (
            "heavy packages must live in extras, not in the base dependencies"
        )


class TestLicenceHygiene:
    COPYLEFT_MARKERS = (
        "gnu general public",
        "gpl-3.0",
        "agpl",
        "copyleft",
        "creativecommons.org/licenses/by-nc",
    )

    # Fingerprints of the widely copied DeepSORT/ByteTrack filter. LOFOP's
    # motion model is written from the estimator equations and must not
    # reintroduce that file's private names or its hand-tuned constants.
    FOREIGN_FINGERPRINTS = (
        "_std_weight_position",
        "_std_weight_velocity",
        "_motion_mat",
        "_update_mat",
        "chi2inv95",
    )

    def test_no_copyleft_markers_in_source(self):
        for path in python_files():
            lowered = path.read_text(encoding="utf-8").lower()
            for marker in self.COPYLEFT_MARKERS:
                assert marker not in lowered, (
                    f"{path.relative_to(PROJECT_ROOT)} contains the copyleft marker "
                    f"{marker!r}; LOFOP ships under Apache-2.0"
                )

    def test_no_foreign_tracker_fingerprints(self):
        for path in python_files():
            text = path.read_text(encoding="utf-8")
            for fingerprint in self.FOREIGN_FINGERPRINTS:
                assert fingerprint not in text, (
                    f"{path.relative_to(PROJECT_ROOT)} contains {fingerprint!r}, a "
                    "naming fingerprint of a GPL-licensed tracker implementation"
                )

    def test_notice_file_exists_and_lists_extras(self):
        notice = PROJECT_ROOT / "NOTICE"
        assert notice.is_file(), "Apache-2.0 projects should ship a NOTICE file"
        text = notice.read_text(encoding="utf-8")
        assert "Apache License" in text
        for package in ("supervision", "numpy", "PyTorch"):
            assert package in text, f"NOTICE does not account for {package}"
