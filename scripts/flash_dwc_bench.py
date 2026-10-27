#!/usr/bin/env python3
"""Cross-check Flash-DWC's reported speedups against the benchmark tables shipped in its source.

Why this script exists
----------------------
The Flash-DWC paper (ACM TACO 2026, 23(4), DOI 10.1145/3837088) reports an average
5.09x speedup for its CUDA-Core GEMV path and 9.01x for its Tensor-Core GEMM path,
both against "the fastest cuDNN DWC algorithm". The paper body is paywalled, so those
two averages cannot be read off a table.

The official repository does, however, ship the raw measurements as inline comments in
four benchmark headers, per GPU and per tile variant:

    performance_dwconv3D_FP16_Gemm_mma.cuh      Flash-DWC-T  (Tensor Core, mma)
    performance_bn_act_dwconv3D_FP16_Gemm.cuh   Flash-DWC    (FP16, CUDA Core GEMV)
    performance_bn_act_dwconv3D_Gemm.cuh        Flash-DWC    (FP32, CUDA Core GEMV)
    cuDNN_performance_dwconv3D_FP16.cuh         cuDNN 9.14.0 baseline, 5 algorithms

This script parses all four and recomputes the speedup distribution, so the paper's two
headline averages can be sanity-checked without the paper.

Reading the result
------------------
`README.md:111-115` warns that the inline comments inside the *kernel* headers are
historical RTX 3060 Ti tuning notes and "should not be interpreted as ... the benchmark
results reported in the paper". The four *performance_* / *cuDNN_performance_* headers
parsed here are a different, more deliberate set: they are organised per GPU (RTX 3060
Ti / RTX 4090 / A100 40GB / A100 80GB), per tile variant, and -- for cuDNN -- per
algorithm, i.e. exactly the axes a paper evaluation table needs. Nothing in the repo
states that these ARE the paper's tables, so treat the output as an independent
consistency check, not as a reproduction of the paper.

Two further caveats the numbers inherit:
  * The cuDNN header records `//We find NCHW format is more efficient for cuDNN`, so the
    baseline is cuDNN's better layout while Flash-DWC runs channel-last. That is the fair
    comparison, and it makes the speedup harder to get, not easier.
  * The CUDA-Core tables come from `bn_act_*` kernels. Whether BN/activation fusion was
    enabled for those runs is not recorded in the comments. The PyTorch entry points
    always pass `<-1, false>`, i.e. fusion off.
"""

import argparse
import math
import os
import re
import sys

# ---------------------------------------------------------------- parsing ----

# GPU is identified by the #ifndef guard that opens each section. A100_80G and
# A100_80GB both occur across files, hence the optional trailing B.
RE_GPU = re.compile(r"#ifndef\s+\w*?(RTX3060TI|RTX4090|A100_80GB?|A100_40GB?)\b")
GPU_LABEL = {
    "RTX3060TI": "RTX 3060 Ti",
    "RTX4090": "RTX 4090",
    "A100_80G": "A100 80GB",
    "A100_80GB": "A100 80GB",
    "A100_40G": "A100 40GB",
    "A100_40GB": "A100 40GB",
}

RE_VARIANT = re.compile(r"^void\s+(\w+)\s*\(")
RE_SHAPE = re.compile(r"\[N,\s*H,\s*W,\s*C\]\s*=\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]")
# A data line either opens a new filter size ("//9 * 9: Size = ..., Time = ..., Performance = ...")
# or continues the previous one ("//    Time = ..., Performance = ... (IGemm)").
RE_ROW = re.compile(
    r"//\s*(?:(\d+)\s*\*\s*(\d+)\s*:.*?)?"
    r"Time\s*=\s*([\d.]+)\s*msec.*?Perfor\w*\s*=\s*([\d.]+)\s*GFlop/s"
    r"(?:\s*\((\w+)\))?"
)

TILE_OF_VARIANT = {
    "8x16": "8x16", "16x8": "16x8", "32x4": "32x4",
    "8_8": "8_8", "8_A": "8_A", "4_8": "4_8", "4_A": "4_A",
}


def variant_of(func_name):
    """Map a harness function name to a short tile label."""
    for pat, label in (("8x16", "8x16"), ("16x8", "16x8"), ("32x4", "32x4"),
                       ("8_8", "8_8"), ("8_A", "8_A"), ("4_8", "4_8"), ("4_A", "4_A")):
        if pat in func_name:
            return label
    return func_name


def parse(path):
    """-> {(gpu, variant, (n, h, w, c)): {fh_fw: [(algo, msec, gflops), ...]}}"""
    out = {}
    gpu = variant = shape = None
    cur_filter = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_GPU.search(line)
            if m and line.lstrip().startswith("#ifndef"):
                gpu = GPU_LABEL[m.group(1)]
                variant = shape = cur_filter = None
                continue
            m = RE_VARIANT.match(line.strip())
            if m:
                variant = variant_of(m.group(1))
                shape = cur_filter = None
                continue
            m = RE_SHAPE.search(line)
            if m:
                shape = tuple(int(x) for x in m.groups())
                cur_filter = None
                continue
            m = RE_ROW.search(line)
            if m and gpu and variant and shape:
                fh, fw, ms, gf, algo = m.groups()
                if fh and fw:
                    cur_filter = (int(fh), int(fw))
                if cur_filter is None:
                    continue
                out.setdefault((gpu, variant, shape), {}).setdefault(cur_filter, []).append(
                    (algo or "-", float(ms), float(gf)))
    return out


def best(rows):
    """Fastest entry of a list of (algo, msec, gflops)."""
    return min(rows, key=lambda r: r[1])


# ------------------------------------------- the dispatcher's tile choice ----

def dispatched_tile(iw):
    """Mirror of elem_..._64_128T_idt's IW-based selection
    (FP16_dwconv3D_Gemm_mma.cuh:19-101) for the common divisible cases."""
    if iw % 32 == 0:
        return "32x4"
    if iw % 16 == 0:
        return "16x8"
    if iw % 8 == 0:
        return "8x16"
    return None          # non-divisible -> main + auxiliary double launch


SOURCES = {
    "tc": "performance_dwconv3D_FP16_Gemm_mma.cuh",
    "c16": "performance_bn_act_dwconv3D_FP16_Gemm.cuh",
    "c32": "performance_bn_act_dwconv3D_Gemm.cuh",
    "cudnn": "cuDNN_performance_dwconv3D_FP16.cuh",
}

# Vendor datasheet figures: FP16 Tensor Core peak *with FP32 accumulate*, dense,
# and peak DRAM bandwidth. The mma the kernel emits is
# mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 (micro_ptx.cuh:113-116), i.e.
# FP32 accumulate, so the FP32-acc column is the correct ceiling -- the GeForce
# FP16-accumulate headline number is 2x this and does not apply.
PEAKS = {
    "RTX 3060 Ti": {"tc": 65.1e12, "bw": 448e9},
    "RTX 4090":    {"tc": 82.6e12, "bw": 1008e9},
    "A100 80GB":   {"tc": 312e12,  "bw": 2039e9},
    "A100 40GB":   {"tc": 312e12,  "bw": 1555e9},
}

# Its RTX 4090 section reports executed throughputs above AD102's own FP32-accumulate
# Tensor-Core peak (129/225 rows, worst 194%), so its absolute numbers are excluded
# from any conclusion drawn here and the speedup statistics are re-run without them.
SUSPECT_GPU = "RTX 4090"

# Toeplitz B-operand MAC density; derivation in flash_dwc_roofline.py.
# 36/64 is only the fw >= 8 limit: a whole 9-tap chunk costs the same 32 mma
# however few of its taps the filter actually uses, so short filters execute
# relatively more zero MACs.
def tc_density(fw):
    """Nonzero (k, n) with 0 <= k - n <= fw - 1 inside the 8x8 mma B window."""
    return sum(min(fw, 8 - n) for n in range(8)) / 64


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/tmp/flashdwc/FlashDWC_cuda/src",
                    help="directory holding the four benchmark headers")
    ap.add_argument("--gpu", default=None, help="restrict to one GPU label")
    ap.add_argument("--all-tiles", action="store_true",
                    help="report the best tile variant instead of the one the dispatcher picks")
    ap.add_argument("--min-filter", type=int, default=3)
    args = ap.parse_args()

    missing = [f for f in SOURCES.values() if not os.path.isfile(os.path.join(args.src, f))]
    if missing:
        print(f"error: benchmark header(s) not found under {args.src}:", ", ".join(missing))
        print("Clone github.com/GilgameshXYZ123/Flash-DWC and pass --src <repo>/FlashDWC_cuda/src")
        return 2

    data = {k: parse(os.path.join(args.src, v)) for k, v in SOURCES.items()}
    for k, v in data.items():
        print(f"parsed {SOURCES[k]:<46} {len(v):>4} (gpu, tile, shape) groups, "
              f"{sum(len(d) for d in v.values()):>5} filter sizes")
    print()

    cudnn = data["cudnn"]
    # cuDNN's harness is not split per tile variant; collapse its variants.
    cudnn_flat = {}
    for (gpu, _var, shape), filters in cudnn.items():
        slot = cudnn_flat.setdefault((gpu, shape), {})
        for f, rows in filters.items():
            slot.setdefault(f, []).extend(rows)

    ratios_tc, ratios_c16 = [], []
    gpus = sorted({g for (g, _s) in cudnn_flat})
    if args.gpu:
        gpus = [g for g in gpus if g == args.gpu]
        if not gpus:
            print(f"error: no cuDNN data for --gpu {args.gpu}; have: {sorted(cudnn_flat)}")
            return 2

    for gpu in gpus:
        shapes = sorted({s for (g, s) in cudnn_flat if g == gpu})
        for shape in shapes:
            n, h, w, c = shape
            tile = dispatched_tile(w)
            tc = data["tc"].get((gpu, tile, shape)) if tile else None
            tc_any = {f: r for (g, v, s), d in data["tc"].items() if g == gpu and s == shape
                      for f, r in d.items()}
            c16 = {f: r for (g, v, s), d in data["c16"].items() if g == gpu and s == shape
                   for f, r in d.items()}
            src = tc_any if args.all_tiles else (tc or {})
            if not src and not c16:
                continue

            print(f"=== {gpu}   [N,H,W,C] = [{n},{h},{w},{c}]   "
                  f"dispatcher tile = {tile or 'main+aux'} ===")
            hdr = (f"{'F':>4} {'ai':>7} | {'cuDNN best':>26} | {'FlashDWC-T':>19} {'x':>6} "
                   f"| {'FlashDWC fp16':>15} {'x':>6}")
            print(hdr)
            print("-" * len(hdr))
            for f in sorted(cudnn_flat[(gpu, shape)]):
                fh, fw = f
                if fw < args.min_filter:
                    continue
                algo, cms, cgf = best(cudnn_flat[(gpu, shape)][f])
                line = f"{fw:>4} {fh*fw/2:>7.1f} | {cgf:>9.1f} GF/s {algo:>7} {cms:>7.3f} ms |"
                if f in src:
                    a2, tms, tgf = best(src[f])
                    sp = tgf / cgf
                    ratios_tc.append((gpu, sp))
                    line += f" {tgf:>9.1f} GF/s {tms:>7.3f} {sp:>6.2f} |"
                else:
                    line += f" {'-':>19} {'-':>6} |"
                if f in c16:
                    a3, cms2, cgf2 = best(c16[f])
                    sp2 = cgf2 / cgf
                    ratios_c16.append((gpu, sp2))
                    line += f" {cgf2:>7.1f} GF/s {sp2:>6.2f}"
                else:
                    line += f" {'-':>15}"
                print(line)
            print()

    def summarise(name, tagged, paper):
        """`tagged` is a list of (gpu, ratio); gm via log-sum so large n cannot overflow."""
        if not tagged:
            print(f"{name}: no overlapping rows")
            return
        xs = sorted(r for _g, r in tagged)
        gm = math.exp(math.fsum(math.log(x) for x in xs) / len(xs))
        print(f"{name}: n={len(xs)}  min={xs[0]:.2f}x  median={xs[len(xs)//2]:.2f}x  "
              f"mean={math.fsum(xs)/len(xs):.2f}x  geomean={gm:.2f}x  max={xs[-1]:.2f}x"
              f"   (paper abstract: {paper})")

    print("=== summary over every (gpu, shape, filter) row present in BOTH harnesses ===")
    summarise("Tensor-Core path  (Flash-DWC-T / cuDNN best)", ratios_tc, "9.01x mean")
    summarise("CUDA-Core FP16 path (Flash-DWC   / cuDNN best)", ratios_c16, "5.09x mean")
    print()
    # The RTX 4090 section reports throughputs above AD102's own Tensor-Core peak
    # (see the reality check below), so re-run both statistics without it.
    if not args.gpu and any(g == SUSPECT_GPU for g, _r in ratios_tc + ratios_c16):
        print(f"=== same statistics with the {SUSPECT_GPU} rows removed ===")
        for name, tagged, paper in (("Tensor-Core path  ", ratios_tc, "9.01x mean"),
                                    ("CUDA-Core FP16 path", ratios_c16, "5.09x mean")):
            summarise(name, [t for t in tagged if t[0] != SUSPECT_GPU], paper)
        print()
    print("note: means here are unweighted over the rows that happen to appear in both")
    print("      harnesses; the paper's averaging window (which filter sizes, which")
    print("      shapes, arithmetic vs geometric mean) is not recoverable without the")
    print("      paper body, so exact agreement is not expected.")
    print()

    # The benchmark headers report *useful* GFlop/s (2*N*OH*OW*C*FH*FW). The Tensor
    # Core actually executes useful/density(fw) MACs, because part of every 64 slots
    # of the Toeplitz B operand are structural zeros. So "% of peak" below is the
    # honest hardware-utilisation figure, and it is what the roofline's "attainable"
    # column should be compared against.
    print("=== peak-efficiency reality check (Tensor-Core path only) ===")
    print("density(fw) = sum_n min(fw, 8-n) / 64 = 0.328 / 0.469 / 0.547 / 0.5625")
    print("              for fw = 3 / 5 / 7 / >=8   (a whole 9-tap chunk costs the")
    print("              same 32 mma however few of its taps the filter uses)")
    print("implied DRAM BW = useful GF/s / AI, AI = FH*FW/2; tile-halo reload is")
    print("ignored, so this UNDERSTATES real traffic and any value > peak is fatal.")
    hdr = (f"{'GPU':<13} {'rows':>5} {'>TC peak':>9} {'>BW peak':>9} "
           f"{'best useful':>13} {'exec TC':>11} {'%TC':>6} {'max%BW':>7}  best row")
    print(hdr)
    tc_gpus = sorted({g for (g, _v, _s) in data["tc"]})
    for gpu in tc_gpus:
        if gpu not in PEAKS:
            continue
        pk = PEAKS[gpu]
        rows = [(gf, v, s, f) for (g, v, s), d in data["tc"].items() if g == gpu
                for f, rl in d.items() for (_a, _ms, gf) in rl]
        if not rows:
            continue
        over_tc = sum(1 for gf, _v, _s, f in rows
                      if gf * 1e9 / tc_density(f[1]) > pk["tc"])
        over_bw = sum(1 for gf, _v, _s, f in rows
                      if gf * 1e9 / (f[0] * f[1] / 2) > pk["bw"])
        max_bw = max(gf * 1e9 / (f[0] * f[1] / 2) / pk["bw"]
                     for gf, _v, _s, f in rows)
        gf, v, s, f = max(rows)
        ex = gf * 1e9 / tc_density(f[1]) / 1e12
        print(f"{gpu:<13} {len(rows):>5} {over_tc:>5}/{len(rows):<3} "
              f"{over_bw:>5}/{len(rows):<3} "
              f"{gf:>8.1f} GF/s {ex:>6.1f} TF/s "
              f"{ex * 1e12 / pk['tc']:>6.1%} {max_bw:>7.1%}  "
              f"{v} [{','.join(map(str, s))}] {f[0]}x{f[1]}")
    print()
    print("Read this as: the density-discounted roofline in flash_dwc_roofline.py is an")
    print("UPPER BOUND. Actual measured efficiency is far below it, so the roofline says")
    print("which regime a shape is in, not what throughput to expect.")
    print()

    # Per-F sweep on each GPU's best shape: exposes both the 9-tap chunk sawtooth
    # (Loop: FH * ((FW + 8) / 9), FP16_dwconv3D_Gemm_mma_kernel_ruse.cuh:15) and the
    # fact that FW=9 has *no* steady-state pipeline iteration at all (:263 loop body
    # runs ceil(FW/9)-1 times), so short filters never reach full cp.async overlap.
    print("=== executed Tensor-Core throughput vs F (best shape per GPU) ===")
    for gpu in tc_gpus:
        if gpu not in PEAKS:
            continue
        rows = [(gf, v, s, f) for (g, v, s), d in data["tc"].items() if g == gpu
                for f, rl in d.items() for (_a, _ms, gf) in rl]
        if not rows:
            continue
        _, bv, bs, _ = max(rows)
        per_f = {}
        for (g, v, s), d in data["tc"].items():
            if g == gpu and v == bv and s == bs:
                for f, rl in d.items():
                    gf = max(r[2] for r in rl)
                    per_f[f[0]] = gf
        print(f"--- {gpu}  {bv} [{','.join(map(str, bs))}]  "
              f"datasheet FP16 TC peak {PEAKS[gpu]['tc'] / 1e12:.1f} TF/s ---")
        print(f"{'F':>4} {'chunks/fh':>10} {'steady':>7} {'useful GF/s':>12} "
              f"{'density':>8} {'exec TF/s':>10} {'%peak':>7}")
        for ff in sorted(per_f):
            gf = per_f[ff]
            ch = (ff + 8) // 9
            d = tc_density(ff)
            ex = gf * 1e9 / d / 1e12
            print(f"{ff:>4} {ch:>10} {ch - 1:>7} {gf:>12.1f} {d:>8.3f} "
                  f"{ex:>10.1f} {ex * 1e12 / PEAKS[gpu]['tc']:>7.1%}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
