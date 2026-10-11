"""GPU kernels of 10 grouped_mm_fp4 calls (torch.profiler, CUDA activity), with whichever FlashInfer is installed."""
import torch
from torch.profiler import ProfilerActivity, profile

import scripts.profile_grouped_sfa_call as p  # builds the operands and warms up

with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for _ in range(10):
        p.call()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="device_time_total", row_limit=8, max_name_column_width=70))
