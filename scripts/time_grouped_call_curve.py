"""Wall time per grouped_mm_fp4 call over 200 consecutive calls (synchronised), reported as medians of 20-call buckets, to show
any warm-up phase. Whichever FlashInfer is installed."""
import json
import time

import torch

import scripts.profile_grouped_sfa_call as p  # operands; it already ran 25 calls

ts = []
for _ in range(200):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    p.call()
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
buckets = [round(sorted(ts[i:i + 20])[10], 3) for i in range(0, 200, 20)]
print(json.dumps({"bucket_median_ms": buckets}))
