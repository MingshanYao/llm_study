#!/usr/bin/env python3
"""Paired ptxas register audit of the Flash-DWC backward-filter (BFC) build.

Backs 表 11 of docs/misc/flash-dwc-tensor-core.html: the dW path ships three
dtype variants, and the only thing that separates `True_FP16_` from `FP16_` at
the same template arguments is the *accumulator* width (half8 + __hfma2 vs
float8 + FP32 SIMD dot). So pairing instances by their template body isolates
the register cost of FP32 accumulation.

ptxas emits, per entry function:

    Compiling entry function '<mangled>' for 'sm_86'
    Function properties for <mangled>
        N bytes stack frame, N bytes spill stores, N bytes spill loads
    Used R registers, S bytes smem, ...

Usage:
    python3 scripts/flash_dwc_regs.py --log /tmp/flashdwc/FlashDWCBFC_cuda/exe/install.log
    python3 scripts/flash_dwc_regs.py --log <path> --by FD     # split by tile width
"""

import argparse
import os
import re
import sys
from collections import defaultdict

RE_FUNC = re.compile(r"Function properties for (\S+)")
RE_USED = re.compile(r"Used (\d+) registers(?:, (\d+) bytes smem)?")
RE_SPILL = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                      r"(\d+) bytes spill loads")
RE_TARGET = re.compile(r"for '(sm_\w+)'")

# Mangled names carry a `_Z<len>` prefix whose digits differ between the
# `True_FP16_` and `FP16_` spellings; strip it before pairing.
RE_ZLEN = re.compile(r"^_Z\d+")

RE_BODY = re.compile(r"^.*?_ruse(.*)$")     # template args live after the family stem
RE_STEM = re.compile(r"^(.*?)_ruse")


def parse(path):
    """-> list of dicts {name, body, stem, regs, smem, spill, target}."""
    out = []
    cur = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = RE_TARGET.search(line)
            if m and "Compiling entry function" in line:
                nm = re.search(r"Compiling entry function '(\S+)'", line)
                cur = {"name": nm.group(1) if nm else None, "target": m.group(1)}
                continue
            m = RE_FUNC.search(line)
            if m:
                if cur is None:
                    cur = {"target": None}
                cur["name"] = m.group(1)
                continue
            m = RE_SPILL.search(line)
            if m and cur is not None:
                cur["stack"] = int(m.group(1))
                cur["spill"] = int(m.group(2)) + int(m.group(3))
                continue
            m = RE_USED.search(line)
            if m and cur is not None and cur.get("name"):
                cur["regs"] = int(m.group(1))
                cur["smem"] = int(m.group(2)) if m.group(2) else 0
                out.append(cur)
                cur = None
    return out


def norm(name):
    """Pair key: mangled name with the `_Z<len>` prefix and the dtype spelling removed."""
    n = RE_ZLEN.sub("", name)
    return n.replace("True_FP16_", "FP16_")


def stem_of(name):
    n = RE_ZLEN.sub("", name)
    for pre in ("True_FP16_", "FP16_", ""):
        if n.startswith(pre):
            n = n[len(pre):]
            break
    m = RE_STEM.match(n)
    return m.group(1) if m else n.split("<")[0]


def fd_of(name):
    """Human label for the tile-width template argument.

    `_ruse`      : <LBX, FD, PAD, CLIP, STAGE_opt, fp32_func_type, BN>
                   (..._dW_GemmSK_kernel_ruse.cuh:35) -> FD is the 2nd arg
    `mega_..._ruse`: <LBX, PAD, CLIP, STAGE_opt, FWr, fp32_func_type, BN>
                   (..._kernel_ruse_mega.cuh:890) -> FD is fixed at 9, the
                   varying arg is FWr (the OW remainder selector, cf. §5.4)
    """
    if "mega_kernel" in name:
        m = re.search(r"_ruseILi\d+ELb\d+ELb\d+ELi\d+ELi(\d+)E", name)
        return f"mega FWr={m.group(1)}" if m else None
    m = re.search(r"_ruseILi\d+ELi(\d+)E", name)
    return f"FD={m.group(1)}" if m else None


def stats(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return {"n": len(xs), "min": xs[0], "med": xs[len(xs) // 2],
            "mean": sum(xs) / len(xs), "max": xs[-1]}


def fmt(s):
    return (f"n={s['n']:<5} min={s['min']:<4} med={s['med']:<4} "
            f"mean={s['mean']:>6.1f} max={s['max']:<4}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="/tmp/flashdwc/FlashDWCBFC_cuda/exe/install.log",
                    help="ptxas build log")
    ap.add_argument("--by", choices=["family", "FD"], default="family",
                    help="group the paired comparison by kernel family stem or by FD")
    ap.add_argument("--list-unpaired", action="store_true",
                    help="print pair keys that appear on only one side")
    args = ap.parse_args()

    if not os.path.isfile(args.log):
        print(f"error: build log not found: {args.log}")
        print("Clone github.com/GilgameshXYZ123/Flash-DWC and point --log at")
        print("FlashDWCBFC_cuda/exe/install.log")
        return 2

    recs = parse(args.log)
    targets = sorted({r["target"] for r in recs if r.get("target")})
    print(f"parsed {len(recs)} ptxas 'Used ... registers' records from {args.log}")
    print(f"compilation targets: {', '.join(targets) or '(none reported)'}")
    spill = [r for r in recs if r.get("spill")]
    stack = [r for r in recs if r.get("stack")]
    print(f"spill (stores+loads > 0): {len(spill)} / {len(recs)}   "
          f"nonzero stack frame: {len(stack)} / {len(recs)}")
    smems = defaultdict(int)
    for r in recs:
        smems[r["smem"]] += 1
    print("static smem per block: "
          + ", ".join(f"{k} B x{v}" for k, v in sorted(smems.items(), key=lambda x: -x[1])))
    print()

    regs_all = stats([r["regs"] for r in recs])
    print(f"ALL kernels        {fmt(regs_all)}")

    # Split by dtype spelling.
    groups = defaultdict(list)
    for r in recs:
        n = RE_ZLEN.sub("", r["name"])
        if n.startswith("True_FP16_"):
            groups["True_FP16 (FP16 accumulate)"].append(r)
        elif n.startswith("FP16_"):
            groups["FP16 (FP32 accumulate)"].append(r)
        elif n.startswith("fp32_") or n.startswith("FP32_"):
            groups["FP32"].append(r)
        else:
            groups["other (utility)"].append(r)
    for k in sorted(groups):
        print(f"{k:<32} {fmt(stats([r['regs'] for r in groups[k]]))}")
    print()

    # Paired comparison: identical template body, only the accumulator differs.
    sides = defaultdict(dict)
    for r in recs:
        n = RE_ZLEN.sub("", r["name"])
        is_true = n.startswith("True_FP16_")
        is_f32acc = n.startswith("FP16_")
        if not (is_true or is_f32acc):
            continue
        key = (norm(r["name"]),
               stem_of(r["name"]) if args.by == "family" else fd_of(r["name"]))
        sides[key]["t" if is_true else "f"] = r["regs"]

    pairs = {k: v for k, v in sides.items() if "t" in v and "f" in v}
    lone = {k: v for k, v in sides.items() if k not in pairs}
    print(f"=== paired comparison ({args.by}): same template body, "
          f"only accumulator dtype differs ===")
    if not pairs:
        print("no pairs found")
        return 1
    print(f"paired keys: {len(pairs)}   unpaired: {len(lone)}")

    buckets = defaultdict(list)
    for (nk, grp), v in pairs.items():
        buckets[grp].append((v["t"], v["f"]))

    print(f"{'group':<44} {'n':>4} {'FP16acc mean':>13} {'range':>10} "
          f"{'FP32acc mean':>13} {'range':>10} {'saving':>7}")
    tot_t, tot_f = [], []

    def gkey(g):
        s = str(g)
        return (s.startswith("mega"), s.split("=")[0],
                int(s.split("=")[1]) if "=" in s and s.split("=")[1].isdigit() else 0)

    for grp in sorted(buckets, key=gkey):
        ts = [t for t, _f in buckets[grp]]
        fs = [f for _t, f in buckets[grp]]
        tot_t += ts
        tot_f += fs
        mt, mf = sum(ts) / len(ts), sum(fs) / len(fs)
        print(f"{str(grp):<44} {len(ts):>4} {mt:>13.1f} "
              f"{str(min(ts)) + '-' + str(max(ts)):>10} {mf:>13.1f} "
              f"{str(min(fs)) + '-' + str(max(fs)):>10} {1 - mt / mf:>7.1%}")
    mt, mf = sum(tot_t) / len(tot_t), sum(tot_f) / len(tot_f)
    print(f"{'TOTAL':<44} {len(tot_t):>4} {mt:>13.1f} "
          f"{str(min(tot_t)) + '-' + str(max(tot_t)):>10} {mf:>13.1f} "
          f"{str(min(tot_f)) + '-' + str(max(tot_f)):>10} {1 - mt / mf:>7.1%}")
    print()
    print(f"FP16 accumulate: {fmt(stats(tot_t))}")
    print(f"FP32 accumulate: {fmt(stats(tot_f))}")
    print(f"register saving  : {(1 - mt / mf):.1%} on the mean, "
          f"{(1 - max(tot_t) / max(tot_f)):.1%} at the worst case "
          f"({max(tot_t)} vs {max(tot_f)} regs)")
    print()
    print("Read this as: on the dW path, True_FP16's win is register pressure, not")
    print("raw throughput -- the FP32-accumulate variant needs ~1.7x the registers")
    print("for the same tile, which is what caps occupancy at large FD.")

    if args.list_unpaired and lone:
        print()
        print(f"--- {len(lone)} unpaired keys ---")
        for (nk, grp), v in sorted(lone.items(), key=lambda x: str(x[0]))[:40]:
            print(f"  {grp}  {'True_FP16' if 't' in v else 'FP16'}  {nk[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
