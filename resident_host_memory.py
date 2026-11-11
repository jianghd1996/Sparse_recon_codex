"""Release already-unused host allocations; never unload live model weights.

Linux diagnostics are optional. No OOM-score or system configuration changes.
"""
import ctypes
import gc
import os
from pathlib import Path


def release_free_host_memory():
    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (OSError, AttributeError):
        pass


def _fields(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            words = value.split()
            if words and words[0].isdigit():
                result[key] = int(words[0])
    return result


def log_host_memory(tag, logger):
    try:
        status = _fields('/proc/self/status')
        mem = _fields('/proc/meminfo')
        pss = _fields('/proc/self/smaps_rollup').get('Pss', 0)
        gib = lambda kb: kb / 1024**2
        logger.info(
            f'[host memory] {tag}: pid={os.getpid()}, ppid={os.getppid()}, '
            f'RSS={gib(status.get("VmRSS", 0)):.2f} GiB, '
            f'PSS={gib(pss):.2f} GiB, '
            f'host_available={gib(mem.get("MemAvailable", 0)):.2f} GiB, '
            f'swap_free={gib(mem.get("SwapFree", 0)):.2f} GiB')
    except (OSError, ValueError):
        pass
