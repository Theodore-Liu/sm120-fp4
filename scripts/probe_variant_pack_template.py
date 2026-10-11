"""Show the exception cuDNN frontend swallows in _variant_pack_uids (when its template preparation fails, the uid order is never
cached and every execute pays for a fresh attempt). Run with whichever FlashInfer is installed."""
import time
import traceback

import torch
import cudnn._pygraph as pg

orig = pg.pygraph._variant_pack_uids if hasattr(pg, "pygraph") else None
cls = None
for name in dir(pg):
    obj = getattr(pg, name)
    if isinstance(obj, type) and hasattr(obj, "_variant_pack_uids"):
        cls = obj
        break
seen = {"n": 0}


def wrapped(self):
    lowered = self._lowered_graph
    if self._sorted_uids is None and lowered is not None and seen["n"] < 2:
        seen["n"] += 1
        try:
            t0 = time.perf_counter()
            lowered._prepare_variant_pack_template()
            print("template ok in", round((time.perf_counter() - t0) * 1e3, 2), "ms; uids", list(lowered._get_variant_pack_uids_sorted())[:12])
        except Exception:  # noqa: BLE001
            print("template FAILED:"); traceback.print_exc(limit=3)
    return cls._orig_vpu(self)


cls._orig_vpu = cls._variant_pack_uids
cls._variant_pack_uids = wrapped
import scripts.profile_grouped_sfa_call  # noqa: E402,F401  (runs the calls)
