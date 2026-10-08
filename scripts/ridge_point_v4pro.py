#!/usr/bin/env python3
"""Roofline analysis of DeepSeek-V4-Pro core operators (prefill vs decode).

Config source: DeepSeek-V4 paper (arXiv:2606.19348) §4.2.1 p.24-25, cross-checked
against official config.json (HF deepseek-ai/DeepSeek-V4-Pro, commit b5968e9).

Accounting conventions (first-order, kernel level):
- AI = FLOPs / HBM bytes moved.
- Weights FP8 (1 B/param), activations BF16 (2 B/el), read once per step.
- KV main entry = 576 B (448 FP8 + 64 RoPE BF16); index entry = 128 B (FP8, inferred).
- Decode: no cross-request reuse; each request's KV + its share of weights read per step.
  Routed-expert weights: an expert's matrix is read once if >=1 token routes to it
  (uniform routing assumed, EPLB-balanced).
- Prefill: flash-style tiling; KV/index blocks amortized over 128-query tiles
  (causal average: a tile sees ~s/8 of the sequence).
- Peaks: dense (no sparsity), public vendor specs.
"""

# ---------------- model config: DeepSeek-V4-Pro ----------------
LAYERS = 61                     # 30 CSA (m=4) + 31 HCA (m'=128), +1 MTP block
D = 7168                        # hidden dim
N_H, C = 128, 512               # query heads x head dim (= KV entry dim)
D_C = 1536                      # q_lora_rank (latent query, shared with indexer)
G, D_G = 16, 1024               # output groups x group rank
N_EXP, K_TOP, I_EXP = 384, 6, 3072   # routed experts, top-k, expert intermediate
N_IH, C_I, TOPK = 64, 128, 1024 # indexer heads / head dim / top-k entries
N_WIN = 128                     # sliding window
V = 129280                      # vocab

# ---------------- precision ----------------
W_B = 1.0      # FP8 weight bytes/param
A_B = 2.0      # BF16 activation bytes/el
KV_MAIN = 576.0
KV_IDX = 128.0

# ---------------- hardware (dense peaks, public specs) ----------------
HW = {
    "H800":  dict(bw=3.35e12, bf16=989e12,  fp8=1979e12),
    "H20":   dict(bw=4.00e12, bf16=148e12,  fp8=296e12),
    "B200":  dict(bw=8.00e12, bf16=2250e12, fp8=4500e12, fp4=9000e12),
    "Rubin": dict(bw=22.0e12, fp4=50e15),   # Vera Rubin GPU, NVFP4; FP8 unpublished
}

def ridge(hw, prec):
    return HW[hw][prec] / HW[hw]["bw"]

# ---------------- projection-GEMM group (per layer) ----------------
# (name, weight params, FLOPs/token, activation bytes/token = 2*(read+write))
# wo_a is group-block-diagonal: 16 groups, each 4096->1024 (NOT dense 65536->16384)
PROJ = [
    ("wq_a",       D * D_C,                  2 * D * D_C,              2 * (D + D_C)),
    ("wq_b",       D_C * N_H * C,            2 * D_C * N_H * C,        2 * (D_C + N_H * C)),
    ("compressor", 4 * D * C,                2 * D * 4 * C,            2 * (D + 4 * C)),
    ("wo_a",       G * (N_H * C // G) * D_G, G * 2 * (N_H * C // G) * D_G, 2 * (N_H * C + G * D_G)),
    ("wo_b",       G * D_G * D,              2 * G * D_G * D,          2 * (G * D_G + D)),
    ("idx_q_b",    D_C * N_IH * C_I,         2 * D_C * N_IH * C_I,     2 * (D_C + N_IH * C_I)),
    ("idx_w",      D * N_IH,                 2 * D * N_IH,             2 * (D + N_IH)),
    ("router",     D * N_EXP,                2 * D * N_EXP,            2 * (D + N_EXP)),
]
P_PROJ = sum(p for _n, p, _f, _a in PROJ)       # weight params/layer
F_PROJ = sum(f for _n, _p, f, _a in PROJ)       # FLOPs/token/layer
T_PROJ = sum(a for _n, _p, _f, a in PROJ)       # act bytes/token/layer

P_SHARED = 3 * D * I_EXP
P_EXPERT = 3 * D * I_EXP
P_LMHEAD = D * V

total = LAYERS * (P_PROJ + P_SHARED + N_EXP * P_EXPERT) + P_LMHEAD
active = LAYERS * (P_PROJ + P_SHARED + K_TOP * P_EXPERT) + P_LMHEAD
print(f"per-layer: proj+router {P_PROJ/1e6:.1f}M  shared {P_SHARED/1e6:.1f}M  "
      f"routed x384 {P_EXPERT * N_EXP/1e9:.2f}B")
print(f"param check: total {total/1e12:.3f}T (paper ~1.6T)   "
      f"active {active/1e9:.2f}B (paper ~49B)")

def proj_ai(b_tok):
    return b_tok * F_PROJ / (P_PROJ * W_B + b_tok * T_PROJ)

def gemm_ai(b_tok, d_in, d_out, w_b=W_B, a_b=A_B):
    """Single weight GEMM: weights read once, activations once per token."""
    return 2 * b_tok * d_in * d_out / (d_in * d_out * w_b + b_tok * (d_in + d_out) * a_b)

# ---------------- attention kernels ----------------
def csa_decode_ai(s):
    """CSA layer, decode, per query token: indexer scores all s/4 entries,
    main attention over top-k=1024 selected + 128 window entries."""
    e_idx, e_sel = s / 4, TOPK + N_WIN
    f = 2 * N_IH * C_I * e_idx + 4 * N_H * C * e_sel
    b = e_idx * KV_IDX + e_sel * KV_MAIN
    return f / b

def hca_decode_ai(s):
    """HCA layer, decode: dense over all s/128 entries + window; no indexer."""
    e = s / 128 + N_WIN
    return 4 * N_H * C * e / (e * KV_MAIN)

def csa_prefill_ai(s, tile=128):
    """CSA layer, prefill: flash tiling; per tile, index+main entries read once
    (union of 128 queries' top-k ~ all available, causal avg s/8)."""
    n_tiles = s / tile
    f = s * (2 * N_IH * C_I * (s / 8) + 4 * N_H * C * (TOPK + N_WIN))
    b = n_tiles * ((s / 8) * (KV_IDX + KV_MAIN))
    return f / b

def hca_prefill_ai(s, tile=128):
    n_tiles = s / tile
    e_avg = s / 256 + N_WIN          # causal avg of cumulative s/128 entries + window
    f = s * 4 * N_H * C * e_avg
    b = n_tiles * e_avg * KV_MAIN
    return f / b

# ---------------- MoE / LM head ----------------
def moe_ai(b_tok, w_b=W_B):
    """Routed experts: b_e = b_tok*K/N tokens per expert (uniform), all-hit weight
    streaming, activations once per assignment."""
    f = b_tok * K_TOP * 6 * D * I_EXP
    b = N_EXP * P_EXPERT * w_b + b_tok * K_TOP * 2 * (D + 2 * I_EXP)
    return f / b

def lmhead_ai(b_tok):
    return gemm_ai(b_tok, D, V)

# ---------------- scenarios ----------------
S, B_REQ, L_PRE = 32768, 150, 32768

print("\n=== ridge points (FLOP/Byte) ===")
for hw in HW:
    print(f"{hw:6s} bw {HW[hw]['bw']/1e12:5.2f} TB/s : " +
          "  ".join(f"{p} {ridge(hw, p):7.0f}" for p in HW[hw] if p != "bw"))

print(f"\n=== DECODE  B={B_REQ} req, s=32K ===")
for tau, tag in [(1, "MTP-1  "), (5, "DSpark ")]:
    b_tok = B_REQ * tau
    rows = [
        ("proj GEMMs", proj_ai(b_tok)),
        ("CSA attn",   csa_decode_ai(S)),
        ("HCA attn",   hca_decode_ai(S)),
        ("routed MoE", moe_ai(b_tok)),
        ("shared exp", gemm_ai(b_tok, D, 3 * I_EXP)),
        ("LM head",    lmhead_ai(b_tok)),
    ]
    print(f"-- {tag} B_tok={b_tok} --")
    for n, ai in rows:
        def v(r): return "COMPUTE" if ai > r else "memory"
        print(f"   {n:11s} AI {ai:8.0f}   H800-FP8 {v(ridge('H800','fp8')):7s}"
              f"   B200-FP8 {v(ridge('B200','fp8')):7s}"
              f"   Rubin-FP4 {v(ridge('Rubin','fp4'))}")

print(f"\n=== PREFILL  L={L_PRE}, 1 request ===")
rows = [
    ("proj GEMMs", proj_ai(L_PRE)),
    ("CSA attn",   csa_prefill_ai(L_PRE)),
    ("HCA attn",   hca_prefill_ai(L_PRE)),
    ("routed MoE", moe_ai(L_PRE)),
    ("shared exp", gemm_ai(L_PRE, D, 3 * I_EXP)),
    ("LM head",    lmhead_ai(L_PRE)),
]
for n, ai in rows:
    def v(r): return "COMPUTE" if ai > r else "memory"
    print(f"   {n:11s} AI {ai:10.0f}   H800-FP8 {v(ridge('H800','fp8')):7s}"
          f"   B200-FP8 {v(ridge('B200','fp8')):7s}"
          f"   Rubin-FP4 {v(ridge('Rubin','fp4'))}")

# ---------------- flip points: B_tok where AI crosses a ridge ----------------
def flip(fn, r, lo=1.0, hi=1e7):
    if fn(hi) < r:
        return None
    for _ in range(200):
        mid = (lo * hi) ** 0.5
        if fn(mid) < r:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5

print("\n=== flip points B_tok (compute-bound above) ===")
for hw, p in [("H800", "fp8"), ("B200", "fp8"), ("B200", "fp4"), ("Rubin", "fp4")]:
    r = ridge(hw, p)
    vals = [
        ("proj GEMMs", flip(proj_ai, r)),
        ("routed MoE", flip(moe_ai, r)),
        ("shared exp", flip(lambda b: gemm_ai(b, D, 3 * I_EXP), r)),
        ("LM head",    flip(lmhead_ai, r)),
    ]
    print(f"{hw}-{p} ridge {r:6.0f}: " +
          "  ".join(f"{n} {'>1e7' if x is None else f'{x:.0f}'}" for n, x in vals))

# ---------------- decode per-step byte breakdown (aggregate, all layers) ----------------
print(f"\n=== decode bytes/step, B={B_REQ}, s=32K, MTP-1 (aggregate over EP fleet) ===")
b_tok = B_REQ
hit = N_EXP * (1 - (1 - K_TOP / N_EXP) ** b_tok)
parts = {
    "routed expert weights": LAYERS * hit * P_EXPERT * W_B,
    "proj+router weights":   LAYERS * P_PROJ * W_B,
    "shared expert weights": LAYERS * P_SHARED * W_B,
    "KV cache (CSA+HCA)":    LAYERS and 30 * b_tok * (S / 4 * KV_IDX + (TOPK + N_WIN) * KV_MAIN)
                             + 31 * b_tok * (S / 128 + N_WIN) * KV_MAIN,
    "LM head weights":       P_LMHEAD * W_B,
    "activations (proj)":    LAYERS * b_tok * T_PROJ,
}
tot = sum(parts.values())
for n, v in sorted(parts.items(), key=lambda kv: -kv[1]):
    print(f"   {n:24s} {v/1e12:7.3f} TB  {100*v/tot:5.1f}%")
print(f"   {'TOTAL':24s} {tot/1e12:7.3f} TB/step")

# ---------------- worked examples: exact intermediates per operator ----------------
print("\n=== worked examples (FLOPs / weights / act bytes -> AI) ===")
for b, tag in [(150, "decode MTP-1 "), (750, "decode DSpark"), (L_PRE, "prefill        ")]:
    print(f"-- {tag} B_tok={b}")
    print(f"   proj   F={b*F_PROJ/1e9:8.2f}G  W={P_PROJ/1e9:6.3f}G  act={b*T_PROJ/1e9:7.3f}G"
          f"  AI={proj_ai(b):8.1f}")
    print(f"   shared F={2*b*D*3*I_EXP/1e9:8.2f}G  W={P_SHARED/1e9:6.3f}G"
          f"  act={b*2*(D+3*I_EXP)/1e9:7.3f}G  AI={gemm_ai(b, D, 3*I_EXP):8.1f}")
    print(f"   lmhead F={2*b*D*V/1e9:8.2f}G  W={P_LMHEAD/1e9:6.3f}G"
          f"  act={b*2*(D+V)/1e9:7.3f}G  AI={lmhead_ai(b):8.1f}")
    print(f"   moe    F={b*K_TOP*6*D*I_EXP/1e9:8.2f}G  W={N_EXP*P_EXPERT/1e9:6.3f}G"
          f"  act={b*K_TOP*2*(D+2*I_EXP)/1e9:7.3f}G  AI={moe_ai(b):8.1f}")
print(f"   proj asymptote F/T = {F_PROJ/T_PROJ:.0f}")
print(f"   csa attn  s=32K: idxF={2*N_IH*C_I*(S/4)/1e6:.1f}M mainF={4*N_H*C*(TOPK+N_WIN)/1e6:.1f}M"
      f" idxB={S/4*KV_IDX/1e6:.3f}M mainB={(TOPK+N_WIN)*KV_MAIN/1e6:.3f}M"
      f"  AI={csa_decode_ai(S):.1f}")
print(f"   hca attn  s=32K: E={S/128+N_WIN:.0f}  F={4*N_H*C*(S/128+N_WIN)/1e6:.1f}M"
      f"  B={(S/128+N_WIN)*KV_MAIN/1e3:.0f}KB  AI={hca_decode_ai(S):.1f}")
print(f"   csa prefill: F={L_PRE*(2*N_IH*C_I*(L_PRE/8)+4*N_H*C*(TOPK+N_WIN))/1e12:.3f}T"
      f"  B={(L_PRE/128)*(L_PRE/8)*(KV_IDX+KV_MAIN)/1e9:.3f}G  AI={csa_prefill_ai(L_PRE):.0f}")
print(f"   hca prefill: F={L_PRE*4*N_H*C*(L_PRE/256+N_WIN)/1e12:.3f}T"
      f"  B={(L_PRE/128)*(L_PRE/256+N_WIN)*KV_MAIN/1e9:.3f}G  AI={hca_prefill_ai(L_PRE):.0f}")
print(f"   moe hits @150: {N_EXP*(1-(1-K_TOP/N_EXP)**150):.0f}/384   @750: "
      f"{N_EXP*(1-(1-K_TOP/N_EXP)**750):.1f}/384")
