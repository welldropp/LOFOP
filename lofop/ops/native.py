"""Build and load the native C++ ops library.

The C++ core is optional by design: every op in :mod:`lofop.ops` has a pure
Python reference implementation (always available, works on every platform),
and the native library is a drop-in accelerator loaded through ctypes. LOFOP
therefore runs everywhere, and *additionally* gets 20-200x faster box ops
when a C++ toolchain is present and the library is built::

    python -c "from lofop.ops.native import build_native; build_native()"

Toolchains supported, tried in this order (override with ``$CXX`` or the
``compiler=`` argument):

* Linux / macOS: ``g++``, ``clang++``, ``c++``
* Windows: ``g++`` / ``clang++`` (MinGW-w64 or LLVM), then MSVC ``cl.exe``
  (run from a Developer prompt so ``cl`` is on PATH)

The compiled library lands next to this module (``_lofop_ops.so`` on
Unix, ``_lofop_ops.dll`` on Windows), falling back to ``~/.cache/lofop`` when
the package directory is read-only, and is picked up automatically next run.
:func:`backend` reports which implementation is active.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from lofop.core.exceptions import LofopError
from lofop.core.logging import get_logger

_LIB_STEM = "_lofop_ops"
_CUDA_LIB_STEM = "_lofop_ops_cuda"
_CSRC = Path(__file__).resolve().parent.parent / "csrc"
_SOURCE = _CSRC / "box_ops.cpp"
# Every translation unit compiled into the native library. Kept as a list so
# new kernels are added by appending here; the C++ SDK links the same files.
_SOURCES = [_SOURCE, _CSRC / "preprocess.cpp"]
_CUDA_SOURCE = _CSRC / "box_ops.cu"
_IS_WINDOWS = sys.platform.startswith("win")

logger = get_logger(__name__)
_loaded: ctypes.CDLL | None = None
_load_attempted = False
_cuda_loaded: ctypes.CDLL | None = None
_cuda_load_attempted = False


def _lib_suffix() -> str:
    """Loadable-library suffix for ctypes on this platform."""
    return ".dll" if _IS_WINDOWS else ".so"


def _candidate_paths(stem: str = _LIB_STEM) -> list[Path]:
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lofop"
    suffix = _lib_suffix()
    return [
        Path(__file__).resolve().parent / f"{stem}{suffix}",
        cache_dir / f"{stem}{suffix}",
    ]


def _compiler_candidates(explicit: str | None) -> list[str]:
    """Ordered list of compiler binaries to try (only those on PATH)."""
    if explicit:
        return [explicit]
    env = os.environ.get("CXX")
    if env:
        return [env]
    if _IS_WINDOWS:
        preference = ["g++", "clang++", "cl"]
    else:
        preference = ["g++", "clang++", "c++"]
    return [name for name in preference if shutil.which(name)] or preference


def _build_command(
    compiler: str, source: Path | list[Path], target: Path
) -> tuple[list[str], Path | None]:
    """Build the compile command for a toolchain.

    Args:
        compiler: Compiler binary.
        source: One source file, or a list of them compiled into one library.
        target: Output library path.

    Returns the argv plus an optional working directory (MSVC scatters
    intermediate .obj/.lib files into cwd, so it runs in a temp dir).
    """
    sources = [source] if isinstance(source, Path) else list(source)
    # Separator-agnostic basename so a Windows "C:\\VS\\cl.exe" path is
    # recognized even when parsed on a POSIX host (and vice versa).
    base = compiler.replace("\\", "/").rsplit("/", 1)[-1].lower()
    name = base[:-4] if base.endswith(".exe") else base
    if name == "cl":  # MSVC
        workdir = Path(tempfile.mkdtemp(prefix="lofop_build_"))
        cmd = [
            compiler, "/nologo", "/O2", "/std:c++17", "/EHsc", "/LD",
            *[str(path) for path in sources], f"/Fe:{target}", f"/Fo:{workdir}\\",
        ]
        return cmd, workdir
    # GNU/Clang-style front end (g++, clang++, c++, MinGW g++).
    cmd = [compiler, "-O3", "-std=c++17", "-shared"]
    if _IS_WINDOWS:
        # MinGW: statically link the runtimes so the DLL has no libgcc/
        # libstdc++ load-time dependency.
        cmd += ["-static-libgcc", "-static-libstdc++"]
    else:
        cmd.append("-fPIC")
    cmd += [str(path) for path in sources]
    cmd += ["-o", str(target)]
    return cmd, None


def build_native(*, force: bool = False, compiler: str | None = None,
                 cuda: bool = False) -> Path:
    """Compile the native ops library and return its path.

    Args:
        force: Rebuild even if a library already exists.
        compiler: Compiler binary to use. Defaults to ``$CXX`` or the first
            available of the platform's preferred compilers.
        cuda: Build the optional CUDA tier from ``box_ops.cu`` instead of the
            C++ tier (requires ``nvcc`` on PATH). The CUDA library is used
            automatically on hosts with a usable NVIDIA GPU and ignored
            everywhere else.

    Raises:
        LofopError: If no compiler is available or every attempt fails. The
            pure Python ops keep working regardless -- building is optional.
    """
    if cuda:
        return _build_cuda(force=force, compiler=compiler)
    existing = find_library()
    if existing is not None and not force:
        return existing

    candidates = _compiler_candidates(compiler)
    errors: list[str] = []
    for binary in candidates:
        if shutil.which(binary) is None and compiler is None and not os.environ.get("CXX"):
            continue
        for target in _candidate_paths():
            target.parent.mkdir(parents=True, exist_ok=True)
            cmd, workdir = _build_command(binary, _SOURCES, target)
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=workdir)
            except FileNotFoundError:
                errors.append(f"{binary}: not found")
                break  # try next compiler, not next path
            except (subprocess.CalledProcessError, OSError) as exc:
                errors.append(f"{binary}: {getattr(exc, 'stderr', exc)}")
                continue
            finally:
                if workdir is not None:
                    shutil.rmtree(workdir, ignore_errors=True)
            logger.info("Built native ops library at %s (via %s)", target, binary)
            _reset_cache()
            return target

    raise LofopError(
        "Could not build the native ops library; pure Python ops remain available. "
        "Install a C++ toolchain (g++/clang++, or MSVC on Windows) or set CXX.",
        context={"tried": candidates, "errors": errors},
    )


def _build_cuda(*, force: bool, compiler: str | None) -> Path:
    existing = find_library(_CUDA_LIB_STEM)
    if existing is not None and not force:
        return existing
    nvcc = compiler or os.environ.get("LOFOP_NVCC") or "nvcc"
    if shutil.which(nvcc) is None:
        raise LofopError(
            "nvcc not found; the CUDA ops tier needs the NVIDIA CUDA toolkit. "
            "The C++ and Python ops keep working regardless -- CUDA is optional.",
            context={"tried": nvcc},
        )
    errors: list[str] = []
    for target in _candidate_paths(_CUDA_LIB_STEM):
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = [nvcc, "-O3", "-std=c++17", "--shared", str(_CUDA_SOURCE), "-o", str(target)]
        if not _IS_WINDOWS:
            cmd[4:4] = ["-Xcompiler", "-fPIC"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            errors.append(f"{nvcc}: {getattr(exc, 'stderr', exc)}")
            continue
        logger.info("Built CUDA ops library at %s (via %s)", target, nvcc)
        _reset_cache()
        return target
    raise LofopError(
        "Could not build the CUDA ops library; the C++ and Python ops remain available.",
        context={"errors": errors},
    )


def find_library(stem: str = _LIB_STEM) -> Path | None:
    """Return the path of a previously built library, if any."""
    for path in _candidate_paths(stem):
        if path.is_file():
            return path
    return None


def load_native() -> ctypes.CDLL | None:
    """Load the native library, or return ``None`` when unavailable.

    The result is cached; call :func:`build_native` (which resets the cache)
    to pick up a fresh build in the same process.
    """
    global _loaded, _load_attempted
    if _load_attempted:
        return _loaded
    _load_attempted = True
    path = find_library()
    if path is None:
        logger.debug("Native ops library not built; using pure Python ops")
        return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        logger.warning("Failed to load native ops library %s: %s", path, exc)
        return None
    lib.lofop_iou_matrix.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.lofop_iou_matrix.restype = None
    lib.lofop_nms.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_float, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.lofop_nms.restype = ctypes.c_int32
    # Kernels added after 1.1.x; hasattr guards keep an older compiled library
    # loadable (its ops still accelerate; the new ops use the Python path).
    if hasattr(lib, "lofop_soft_nms"):
        lib.lofop_soft_nms.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
        ]
        lib.lofop_soft_nms.restype = ctypes.c_int32
    if hasattr(lib, "lofop_decode_dense"):
        lib.lofop_decode_dense.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.lofop_decode_dense.restype = ctypes.c_int32
    if hasattr(lib, "lofop_letterbox"):
        lib.lofop_letterbox.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        lib.lofop_letterbox.restype = ctypes.c_int32
    if hasattr(lib, "lofop_unletterbox_boxes"):
        lib.lofop_unletterbox_boxes.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ]
        lib.lofop_unletterbox_boxes.restype = ctypes.c_int32
    _loaded = lib
    logger.debug("Loaded native ops library from %s", path)
    return lib


def load_native_cuda() -> ctypes.CDLL | None:
    """Load the CUDA ops library, or ``None`` when unavailable or no GPU.

    Only returns a library when a CUDA device is actually usable at load
    time, so callers can treat "loaded" as "safe to dispatch to".
    """
    global _cuda_loaded, _cuda_load_attempted
    if _cuda_load_attempted:
        return _cuda_loaded
    _cuda_load_attempted = True
    path = find_library(_CUDA_LIB_STEM)
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        logger.warning("Failed to load CUDA ops library %s: %s", path, exc)
        return None
    lib.lofop_cuda_available.argtypes = []
    lib.lofop_cuda_available.restype = ctypes.c_int32
    if lib.lofop_cuda_available() != 1:
        logger.debug("CUDA ops library built but no usable CUDA device; ignoring")
        return None
    lib.lofop_cuda_iou_matrix.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.lofop_cuda_iou_matrix.restype = ctypes.c_int32
    lib.lofop_cuda_decode_dense.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.lofop_cuda_decode_dense.restype = ctypes.c_int32
    _cuda_loaded = lib
    logger.debug("Loaded CUDA ops library from %s", path)
    return lib


def backend() -> str:
    """Return the fastest active ops tier: ``"cuda"``, ``"native"`` (C++),
    or ``"python"``. Tiers stack -- ``"cuda"`` implies the lower tiers remain
    available and are used for the ops CUDA does not cover."""
    if load_native_cuda() is not None:
        return "cuda"
    return "native" if load_native() is not None else "python"


def _reset_cache() -> None:
    global _loaded, _load_attempted, _cuda_loaded, _cuda_load_attempted
    _loaded = None
    _load_attempted = False
    _cuda_loaded = None
    _cuda_load_attempted = False
