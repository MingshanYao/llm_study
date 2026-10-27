#!/usr/bin/env python3
"""Roofline analysis of depthwise convolution (DWC) and of Flash-DWC's Tensor-Core mapping.

Why this script exists
----------------------
Flash-DWC (ACM TACO 2026, 23(4), DOI 10.1145/3837088) claims that DWC becomes
compute-efficient once its per-channel GEMVs are re-expressed as compact
high-density GEMMs on Tensor Cores. This script derives, from first principles,
the two numbers that decide whether that claim can hold:

  (1) the arithmetic intensity (AI) of DWC as a closed form, and
  (2) the Tensor-Core MAC density of Flash-DWC's Toeplitz B-operand.

Every structural constant below is read off the official CUDA source
(github.com/GilgameshXYZ123/Flash-DWC, Apache-2.0), cited as path:line.

Accounting conventions (first-order, kernel level)
--------------------------------------------------
- AI = useful FLOPs / HBM bytes moved. "Useful" = the 2*N*OH*OW*C*FH*FW MACs
  the convolution semantically requires; padding zeros that the Tensor Core
  still multiplies are counted separately as F_exec.
- Forward pass, same-padding stride-1 (the only regime the Tensor-Core kernel
  supports: Torch_Impl_fp16.cuh:148-170 takes no padding/stride argument).
- Activations FP16 (2 B/el), filter FP16. X read once, Y written once.
- Filter traffic is T-independent and negligible: FH*FW*C*2 B against
  4*N*H*W*C B of activation traffic.

Tensor-Core density derivation
------------------------------
Flash-DWC-T issues mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
(micro_ptx.cuh:108-116) with

    A[m][k] = X[pixel-group m, input offset k, channel c]   (m,k in [0,16)x[0,8))
    B[k][n] = W[k - n]                                      (n in [0,8))
    C[m][n] = sum_k A[m][k]*B[k][n] = sum_u W[u]*X[m, u+n] = Y[pixel n of group m]

i.e. the mma's N axis carries 8 consecutive output pixels and its B operand is
the filter's Toeplitz matrix. B is built from a shared-memory buffer Ws[25][9]
whose rows 0..6 and 16..22 are pre-zeroed (FP16_dwconv3D_Gemm_mma_kernel_ruse.cuh:119-120),
and indexed as Ws[7 - n + k] (:140), so B[k][n] == 0 exactly when k < n.

    nonzeros(B) = sum_{n=0..7} (8 - n) = 36   out of 64  ->  density = 56.25%

Cross-check against the source's own instruction count: per warp, per filter
row fh, one 9-tap chunk costs 32 mma (16 in the "L-buffer" stage + 16 in the
"H-buffer" stage; FP16_dwconv3D_Gemm_mma_kernel_ruse.cuh:210-258), each m16n8k8
= 1024 MACs, so 32768 executed MACs against 16 channels * 128 pixels * 9 taps
= 18432 useful MACs -> 18432/32768 = 56.25%. Both routes agree.
"""

import argparse

# ---------------- Tensor-Core mapping constants (from source) ----------------
CTA_CHANNELS = 64        # blockIdx.y = IC>>3>>3      [FP16_dwconv3D_Gemm_mma_kernel_ruse.cuh:27]
CTA_PIXELS = 128         # "64 (IC) * 128 (N*OH*OW)"  [FP16_dwconv3D_Gemm_mma.cuh:13]
BLOCK = (8, 16)          # dim3(8, 16) = 128 threads  [..._kernel_ruse.cuh:28]
WARPS_PER_CTA = 4        # 128 threads / 32
MMA_M, MMA_N, MMA_K = 16, 8, 8          # m16n8k8      [micro_ptx.cuh:108-116]
MMA_CHUNKS_PER_9TAPS = 32               # 16 (L) + 16 (H) per warp per fh per 9-tap chunk
TAPS_PER_CHUNK = 9                      # ofw += 9     [..._kernel_ruse.cuh:263]

TOEPLITZ_NONZERO = sum(8 - n for n in range(8))   # = 36
TOEPLITZ_TOTAL = MMA_K * MMA_N                    # = 64
TC_DENSITY = TOEPLITZ_NONZERO / TOEPLITZ_TOTAL    # = 0.5625


def tc_density(fw):
    """MAC density of the 8x8 B window for a filter of width `fw`.

    A whole 9-tap chunk costs the same 32 mma regardless of how many of those
    9 taps the filter actually uses, so 36/64 is only the fw >= 8 limit.
    Nonzero slots are the (k, n) with 0 <= k - n <= fw - 1:

        count(fw) = sum_{n=0..7} min(fw, 8 - n)

    fw:      1    3    5    7    8+
    count:   8   21   30   35   36
    """
    return sum(min(fw, 8 - n) for n in range(8)) / TOEPLITZ_TOTAL


# Naive alternatives, for contrast (derived, not measured):
#  - per-channel GEMV forced onto mma: the N axis would hold 1 useful column of 8
#  - one big GEMM over a 64-channel tile: block-diagonal in channel, 1/64 useful
DENSITY_GEMV_NAIVE = 1 / MMA_N
DENSITY_BLOCKDIAG_NAIVE = 1 / CTA_CHANNELS

# ---------------- GPU peaks: FP16 Tensor Core, FP32 accumulate, dense --------
# Vendor datasheet figures, not achievable-throughput figures.
GPUS = {
    # NVIDIA A100 datasheet (v1.9): 312 TFLOPS FP16 Tensor w/ FP32 acc, 2039 GB/s HBM2e (SXM4 80GB)
    "a100-sxm":   {"label": "A100 SXM4 80GB",  "fp16_tc": 312e12, "bw": 2039e9, "sm": 80},
    # NVIDIA H100 datasheet: 989 TFLOPS FP16 Tensor w/ FP32 acc, 3.35 TB/s HBM3 (SXM5)
    "h100-sxm":   {"label": "H100 SXM5 80GB",  "fp16_tc": 989e12, "bw": 3.35e12, "sm": 80},
    # NVIDIA Ada whitepaper / RTX 4090 spec: 82.6 TFLOPS FP16 Tensor w/ FP32 acc dense, 1008 GB/s GDDR6X
    "rtx4090":    {"label": "RTX 4090",        "fp16_tc": 82.6e12, "bw": 1008e9, "sm": 89},
    # RTX 3060 Ti (GA104, sm_86): 65.1 TFLOPS FP16 Tensor w/ FP32 acc dense, 448 GB/s GDDR6.
    # Listed only as a cross-check candidate: README.md:111-115 says the inline benchmark
    # comments in the kernels are historical notes from early RTX 3060 Ti tuning runs.
    "rtx3060ti":  {"label": "RTX 3060 Ti",     "fp16_tc": 65.1e12, "bw": 448e9,  "sm": 86},
    # FP32 vector path (the CUDA-Core GEMV kernel), for the memory-bound comparison
    # A100: 19.5 TFLOPS FP32; H100: 67 TFLOPS; RTX 4090: 82.6 TFLOPS FP32
    "a100-fp32":  {"label": "A100 SXM4 (FP32 vector)", "fp16_tc": 19.5e12, "bw": 2039e9, "sm": 80},
    "4090-fp32":  {"label": "RTX 4090 (FP32 vector)",  "fp16_tc": 82.6e12, "bw": 1008e9, "sm": 89},
}

# Realistic achievable fractions, for the "attainable" column only.
BW_EFF = 0.80
TC_EFF = 0.70


def dwc_flops(n, oh, ow, c, fh, fw):
    """Useful FLOPs of a depthwise convolution (2 FLOP per MAC)."""
    return 2 * n * oh * ow * c * fh * fw


def dwc_bytes(n, ih, iw, c, fh, fw, elem=2):
    """HBM traffic: X read once + Y written once + filter (T-independent)."""
    b_act = elem * (n * ih * iw * c) + elem * (n * ih * iw * c)   # OH=IH, OW=IW
    b_w = elem * fh * fw * c
    return b_act + b_w


def ai_useful(fh, fw, elem=2):
    """Closed form. Activation traffic dominates, so N/H/W/C cancel:

        AI = 2*N*H*W*C*FH*FW / (2 * elem * N*H*W*C) = FH*FW / elem
    """
    return fh * fw / elem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="a100-sxm", choices=sorted(GPUS))
    ap.add_argument("--filters", default="3,5,7,9,13,17,21,27,31,45,53",
                    help="comma-separated square filter sizes to tabulate")
    args = ap.parse_args()

    g = GPUS[args.gpu]
    ridge = g["fp16_tc"] / g["bw"]
    # Tensor Core becomes the binding roof when  F_exec/peak > B/bw, i.e.
    #   AI_useful > TC_DENSITY * ridge   ->   FH*FW > 2 * TC_DENSITY * ridge
    ai_cross = TC_DENSITY * ridge
    f_cross = (2 * ai_cross) ** 0.5

    print(f"GPU                        : {g['label']}  (sm_{g['sm']})")
    print(f"FP16 Tensor peak (FP32 acc): {g['fp16_tc']/1e12:.1f} TFLOPS")
    print(f"HBM bandwidth              : {g['bw']/1e12:.3f} TB/s")
    print(f"ridge point                : {ridge:.1f} FLOP/byte")
    print()
    print("DWC arithmetic intensity (closed form, forward, same-padding stride-1)")
    print("  AI_useful = FH*FW / 2   [FLOP/byte]   -- independent of N, H, W, C")
    print("  AI_exec   = TC_DENSITY * AI_useful = 0.5625 * FH*FW / 2")
    print(f"  Tensor-Core MAC density  = {TOEPLITZ_NONZERO}/{TOEPLITZ_TOTAL} = {TC_DENSITY:.4f}")
    print(f"    (naive per-channel GEMV on mma : {DENSITY_GEMV_NAIVE:.4f})")
    print(f"    (naive block-diagonal GEMM     : {DENSITY_BLOCKDIAG_NAIVE:.4f})")
    print(f"  density gain vs GEMV-naive       : {TC_DENSITY/DENSITY_GEMV_NAIVE:.1f}x")
    print(f"  density gain vs block-diagonal   : {TC_DENSITY/DENSITY_BLOCKDIAG_NAIVE:.1f}x")
    print(f"  compute-bound once AI_useful > {ai_cross:.1f}  ->  square filter F > {f_cross:.1f}")
    print()

    hdr = (f"{'F x F':>7} {'taps':>5} {'AI_use':>8} {'AI/ridge':>9} "
           f"{'bound':>10} {'attain TF/s':>12} {'mma/warp/fh':>12}")
    print(hdr)
    print("-" * len(hdr))
    for f in [int(x) for x in args.filters.split(",")]:
        fh = fw = f
        ai = ai_useful(fh, fw)
        ratio = ai / ridge
        bound = "MEM" if ratio < 0.5 else ("COMPUTE" if ratio > 2.0 else "BALANCED")
        # attainable useful throughput = min(roof * efficiency), density-discounted
        t_compute = g["fp16_tc"] * TC_EFF * TC_DENSITY
        t_mem = g["bw"] * BW_EFF * ai
        attain = min(t_compute, t_mem) / 1e12
        mma_per_chunk = MMA_CHUNKS_PER_9TAPS * ((fw + TAPS_PER_CHUNK - 1) // TAPS_PER_CHUNK)
        print(f"{f:>3}x{f:<3} {fh*fw:>5} {ai:>8.1f} {ratio:>9.2f} "
              f"{bound:>10} {attain:>12.1f} {mma_per_chunk:>11}")

    print()
    print("Cross-check against inline benchmark comments in the source")
    print("  shape [N,H,W,C] = [128,64,64,256], FP16")
    print("  (FP16_dwconv3D_Gemm_mma_kernel_ruse.cuh:76-96; README.md:113 flags these")
    print("   as historical tuning notes, NOT the paper's reported results)")
    print("  The comments never state which GPU produced them, so 'TC TFLOPS' is")
    print("  reported as % of three candidates' FP16 Tensor peak. Only a value")
    print("  <= 100% is physically possible; README.md:111-115 points at RTX 3060 Ti.")
    bench = [(5, 1.663, 4034.8), (7, 2.082, 6319.0), (9, 2.586, 8406.7),
             (13, 5.725, 7924.2), (17, 7.486, 10362.5), (21, 12.070, 9808.0),
             (27, 15.640, 12512.3), (31, 22.425, 11503.6), (45, 37.459, 14511.3),
             (53, 50.744, 14859.6)]
    n, h, w, c = 128, 64, 64, 256
    cand = [(k, GPUS[k]) for k in ("rtx3060ti", "a100-sxm", "rtx4090")]
    print(f"  {'F':>3} {'msec':>8} {'src GF/s':>9} {'calc GF/s':>10} {'AI':>7} "
          f"{'dens':>6} {'TC TFLOPS':>10} "
          + " ".join(f"{'%' + k:>11}" for k, _ in cand))
    for f, ms, reported in bench:
        fl = dwc_flops(n, h, w, c, f, f)
        calc = fl / (ms * 1e-3) / 1e9
        d = tc_density(f)
        tc = calc / d / 1e3      # executed Tensor-Core TFLOPS
        row = " ".join(f"{tc * 1e12 / g_['fp16_tc']:>10.1%}" for _, g_ in cand)
        print(f"  {f:>3} {ms:>8.3f} {reported:>9.1f} {calc:>10.1f} "
              f"{ai_useful(f, f):>7.1f} {d:>6.3f} {tc:>10.1f} {row}")


if __name__ == "__main__":
    main()
