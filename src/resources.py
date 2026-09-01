"""Resource guards. Import this FIRST, before numpy, in every entry point.

Two jobs:
  1. Cap BLAS/OpenMP threads before numpy loads, so a matrix op cannot saturate every core
     and make the machine unusable.
  2. Refuse allocations that would exhaust RAM, with a clear message naming the culprit,
     instead of letting the OS thrash.

Set FORECAST_THREADS to override the thread cap (default 4 of N cores, leaving headroom).
"""
import os


def _default_threads():
    try:
        n = os.cpu_count() or 4
    except Exception:
        n = 4
    return str(max(1, min(4, n - 2)))


_T = os.environ.get("FORECAST_THREADS", _default_threads())
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _T)

THREADS = int(_T)


def available_ram_bytes():
    """Physical RAM currently available, without requiring psutil."""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.ullAvailPhys)
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * 1024 ** 3          # conservative fallback


def require_ram(n_bytes, what="operation", headroom=0.60):
    """Raise before allocating if `n_bytes` would use more than `headroom` of free RAM."""
    avail = available_ram_bytes()
    if n_bytes > avail * headroom:
        raise MemoryError(
            f"{what} needs {n_bytes/1e9:.2f} GB but only {avail/1e9:.2f} GB is free "
            f"(cap = {headroom:.0%}). Reduce --max-snapshots or the cache limit, or run this step alone."
        )
    return True


def gb(n_bytes):
    return f"{n_bytes/1e9:.2f} GB"


def report(prefix=""):
    print(f"{prefix}threads={THREADS}  free RAM={gb(available_ram_bytes())}", flush=True)
