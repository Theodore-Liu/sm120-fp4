"""Add diff statistics after the TensorEquals check in CUTLASS examples 79a and 79d (run on the pod, cwd = cutlass root)."""
import subprocess

TARGETS = [
    ("examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu",
     "block_reference_D", "block_D", "block_D.size()", False),
    ("examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu",
     "block_ref_D.at(i)", "block_D.at(i)", "block_D.at(i).size()", True),
]
subprocess.run(["git", "checkout", "--"] + [t[0] for t in TARGETS], check=True)

for path, ref, dev, size, grouped in TARGETS:
    text = open(path).read()
    text = text.replace("#include <iostream>", "#include <iostream>\n#include <cmath>\n#include <cstdio>", 1)
    lines = text.split("\n")
    out, done = [], False
    tag = ' group=%d' if grouped else ''
    arg = ', i' if grouped else ''
    for line in lines:
        out.append(line)
        if not done and "TensorEquals(" in line and ref in line:
            out.append(f"  {{ auto* rr = {ref}.host_data(); auto* dd = {dev}.host_data(); size_t nn = {size}; size_t cnt = 0; double mx = 0; double mref = 0;")
            out.append("    for (size_t q = 0; q < nn; ++q) { double a = float(rr[q]), b = float(dd[q]); double df = std::fabs(a - b); if (std::fabs(a) > mref) mref = std::fabs(a);")
            out.append(f"      if (df > 0) {{ if (cnt < 4) std::printf(\"  DIFF{tag}[%zu] ref=%g got=%g\\n\"{arg}, q, a, b); cnt++; if (df > mx) mx = df; }} }}")
            out.append(f"    std::printf(\"  DIFFSTAT{tag} elements=%zu differing=%zu maxabs=%g refabsmax=%g\\n\"{arg}, nn, cnt, mx, mref); }}")
            done = True
    open(path, "w").write("\n".join(out))
    print(path.split("/")[-1], "patched" if done else "NOT patched")
