#!/usr/bin/env python3
"""Audit the self-consistency of the benchmark tables shipped in Flash-DWC's source.

Why this script exists
----------------------
`flash_dwc_bench.py` recomputes the paper's speedups from the `//F*F: Size = ...,
Time = ..., Performance = ...` comments in three benchmark headers. Those comments are
the only raw measurements the repository ships, so before trusting any ratio derived
from them it is worth asking whether they are internally consistent. The harness that
produced them defines two identities (test_fp16.cuh:451,455,456 and the FP32/BFC twins):

    Size        = 2 * (C/1024) * (N*OH*OW/1024) * (FH*FW/1024)
    Performance = Size * 1024^3 * 1e-9 / (Time * 1e-3)          [decimal GFlop/s]

with OH=H, OW=W because every measured configuration is same-padding stride-1. Both are
recomputable from the shape label printed a few lines above each block, so a row that
disagrees with its own label is a transcription defect, not a measurement.

What it reports
---------------
1. per-file row counts and Size / Performance mismatch counts
2. the fixed-ratio blocks, where a whole 25-row Size ladder is a constant multiple of
   its label -- the signature of a shape label pasted from the neighbouring block
3. rows whose *useful* throughput alone exceeds the GPU's raw FP16 Tensor-Core peak
   (useful <= executed always, so this needs no density model to be fatal)
4. blast radius: which of the defective lines are actually picked as a winner by
   flash_dwc_bench's ratio statistics
5. the CUDA-Core FP16 speedup distribution recomputed three ways -- as recorded, with
   the labels corrected in place, and with the suspect GPU dropped

Only self-contained rows (those carrying both `F*F:` and `Size`) are auditable. The
FP32 header has 150 continuation lines and the cuDNN header 1188; they carry a Time and
a Performance but no Size, so they are counted and skipped.
"""

import argparse
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flash_dwc_bench as B          # noqa: E402  (regexes, GPU labels, PEAKS, tc_density)

RE_SIZE = re.compile(r"Size\s*=\s*([\d.]+)")

SOURCES = OrderedDict((k, v) for k, v in B.SOURCES.items() if k != "cudnn")
CUDNN = B.SOURCES["cudnn"]


class Row:
    __slots__ = ("key", "line", "gpu", "variant", "shape", "fh", "fw",
                 "size", "msec", "gf", "algo")

    def __init__(self, key, line, gpu, variant, shape, fh, fw, size, msec, gf, algo):
        self.key, self.line, self.gpu, self.variant, self.shape = key, line, gpu, variant, shape
        self.fh, self.fw, self.size, self.msec, self.gf, self.algo = fh, fw, size, msec, gf, algo

    @property
    def filter(self):
        return (self.fh, self.fw)


def parse_rows(key, path):
    """Line-numbered row parse.

    Returns every timing row, with `size is None` on continuation lines -- those carry a
    Time, a Performance and an algorithm tag but no Size, so they take part in winner
    selection (cuDNN records one line per algorithm, 1188 of its 1638 rows) but cannot be
    Size-audited.
    """
    rows, cont = [], 0
    gpu = variant = shape = None
    cur = None
    with open(path, "r", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            m = B.RE_GPU.search(line)
            if m and line.lstrip().startswith("#ifndef"):
                gpu, variant, shape, cur = B.GPU_LABEL[m.group(1)], None, None, None
                continue
            m = B.RE_VARIANT.match(line.strip())
            if m:
                variant, shape, cur = B.variant_of(m.group(1)), None, None
                continue
            m = B.RE_SHAPE.search(line)
            if m:
                shape, cur = tuple(int(x) for x in m.groups()), None
                continue
            m = B.RE_ROW.search(line)
            if not m or not (gpu and variant and shape):
                continue
            fh, fw, ms, gf, algo = m.groups()
            if fh and fw:
                cur = (int(fh), int(fw))
            if cur is None:
                continue
            sm = RE_SIZE.search(line)
            if sm is None:
                cont += 1
            rows.append(Row(key, lineno, gpu, variant, shape, cur[0], cur[1],
                            float(sm.group(1)) if sm else None,
                            float(ms), float(gf), algo or "-"))
    return rows, cont


def label_size(shape, fh, fw):
    """Size the harness would print for this shape and filter."""
    n, h, w, c = shape
    return 2.0 * (c / 1024) * (n * h * w / 1024) * (fh * fw / 1024)


def label_perf(size, msec):
    return size * 1024 ** 3 * 1e-9 / (msec * 1e-3)


def summarise(name, ratios, paper):
    if not ratios:
        print(f"{name}: no rows")
        return
    xs = sorted(ratios)
    gm = math.exp(math.fsum(math.log(x) for x in xs) / len(xs))
    print(f"{name:<22} n={len(xs):>4}  min={xs[0]:.2f}x  median={xs[len(xs)//2]:.2f}x  "
          f"mean={math.fsum(xs)/len(xs):.2f}x  geomean={gm:.2f}x  max={xs[-1]:.2f}x"
          f"   (paper abstract: {paper})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/tmp/flashdwc/FlashDWC_cuda/src",
                    help="directory holding the four benchmark headers")
    ap.add_argument("--size-tol", type=float, default=0.02,
                    help="absolute tolerance on the printed Size field")
    ap.add_argument("--perf-tol", type=float, default=0.01,
                    help="relative tolerance on the printed Performance field")
    ap.add_argument("--list-defects", action="store_true",
                    help="print every defective line instead of a per-file summary")
    args = ap.parse_args()

    missing = [f for f in list(SOURCES.values()) + [CUDNN]
               if not os.path.isfile(os.path.join(args.src, f))]
    if missing:
        print(f"error: benchmark header(s) not found under {args.src}: {', '.join(missing)}")
        print("Clone github.com/GilgameshXYZ123/Flash-DWC and pass --src <repo>/FlashDWC_cuda/src")
        return 2

    data, cont = {}, {}
    for key, fn in list(SOURCES.items()) + [("cudnn", CUDNN)]:
        data[key], cont[key] = parse_rows(key, os.path.join(args.src, fn))

    print("=== 1. auditable rows ===")
    print("auditable = carries both `F*F:` and `Size`; continuation lines have no Size")
    for key, fn in list(SOURCES.items()) + [("cudnn", CUDNN)]:
        n = sum(1 for r in data[key] if r.size is not None)
        print(f"  {fn:<46} {n:>5} auditable   {cont[key]:>5} continuation")
    three = [r for k in SOURCES for r in data[k] if r.size is not None]
    print(f"  {'TOTAL (three Flash-DWC headers)':<46} {len(three):>5}")
    print()

    print("=== 2. Size / Performance self-check ===")
    print(f"tolerance: Size +/-{args.size_tol} absolute, Performance +/-{args.perf_tol:.0%} relative")
    defects = defaultdict(list)          # key -> [(row, kind, recorded, recomputed)]
    for r in three:
        want = label_size(r.shape, r.fh, r.fw)
        bad_size = abs(r.size - want) > args.size_tol
        bad_perf = abs(r.gf - label_perf(r.size, r.msec)) > args.perf_tol * label_perf(r.size, r.msec)
        if bad_size:
            defects[r.key].append((r, "Size", r.size, want))
        if bad_perf:
            defects[r.key].append((r, "Perf", r.gf, label_perf(r.size, r.msec)))
    hdr = f"  {'file':<46} {'rows':>5} {'Size bad':>9} {'Perf bad':>9}"
    print(hdr)
    tot_sz = tot_pf = 0
    for key in SOURCES:
        sz = sum(1 for _r, k, _a, _b in defects[key] if k == "Size")
        pf = sum(1 for _r, k, _a, _b in defects[key] if k == "Perf")
        tot_sz += sz
        tot_pf += pf
        n = sum(1 for r in data[key] if r.size is not None)
        print(f"  {SOURCES[key]:<46} {n:>5} {sz:>9} {pf:>9}")
    print(f"  {'TOTAL':<46} {len(three):>5} {tot_sz:>5} ({tot_sz/len(three):.1%}) "
          f"{tot_pf:>5} ({tot_pf/len(three):.1%})")
    print()

    if args.list_defects:
        print("=== 2b. every defective line ===")
        for key in SOURCES:
            for r, kind, got, want in sorted(defects[key], key=lambda d: d[0].line):
                print(f"  {SOURCES[key]}:{r.line}  {r.gpu} {r.variant} "
                      f"[{','.join(map(str, r.shape))}] {r.fh}x{r.fw}  "
                      f"{kind}: recorded {got} vs expected {want:.2f}  (ratio {got/want:.3f})")
        print()

    print("=== 3. fixed-ratio blocks ===")
    print("a whole Size ladder off by one constant multiple = the shape label was pasted")
    print("from a neighbouring block (or the ladder was). Ratio shown is recorded/label.")
    print("the span below covers the DEFECTIVE DATA ROWS; the block's shape label is the")
    print("line above the first one and its closing brace the line after the last.")
    blocks = defaultdict(list)
    for key in SOURCES:
        for r, kind, got, want in defects[key]:
            if kind == "Size":
                blocks[(key, r.gpu, r.variant, r.shape)].append((r.line, got / want))
    for (key, gpu, variant, shape), rs in sorted(blocks.items(), key=lambda kv: -len(kv[1])):
        ratios = [x for _l, x in rs]
        lo, hi, med = min(ratios), max(ratios), ratios[len(ratios) // 2]
        lines = [l for l, _x in rs]
        span = f"{min(lines)}-{max(lines)}" if len(lines) > 1 else str(lines[0])
        fixed = "FIXED ratio" if hi - lo < 0.02 else f"varies {lo:.3f}..{hi:.3f}"
        print(f"  {len(rs):>3} rows  {SOURCES[key]}:{span:<10} {gpu:<12} {variant:<8} "
              f"[{','.join(map(str, shape))}]  median x{med:.3f}  {fixed}")
    print()

    print("=== 4. rows whose USEFUL throughput alone exceeds the raw FP16 TC peak ===")
    print("useful <= executed always, so no Toeplitz density model is needed here.")
    print(f"{'file':<44} {'line':>5} {'GPU':<12} {'tile':<6} {'F':>3} {'msec':>8} "
          f"{'GF/s':>9} {'%peak':>7}")
    worst = 0.0
    for r in data["tc"]:
        pk = B.PEAKS.get(r.gpu)
        if not pk:
            continue
        pct = r.gf * 1e9 / pk["tc"]
        if pct > 1.0:
            worst = max(worst, pct)
            print(f"  {SOURCES['tc']:<42} {r.line:>5} {r.gpu:<12} {r.variant:<6} "
                  f"{r.fw:>3} {r.msec:>8.3f} {r.gf:>9.1f} {pct:>7.1%}")
    print(f"  worst = {worst:.1%} of the datasheet FP32-accumulate peak")
    print("  adding the density discount (56.25% for fw>=8) raises the violation count;")
    print("  see flash_dwc_bench.py's peak-efficiency reality check for that basis.")
    print()

    # ---- winner provenance, replicating flash_dwc_bench's selection rule ----
    def winners(key, dispatched=False):
        """(gpu, shape, filter) -> winning Row, chosen by min msec as B.best() does.

        `dispatched` mirrors flash_dwc_bench's default for the Tensor-Core path: only the
        tile variant the library's dispatcher would pick for that width is considered, so
        the ratio reflects what a user actually runs rather than the best of three tiles.
        """
        groups = defaultdict(list)
        for r in data[key]:
            if dispatched and r.variant != B.dispatched_tile(r.shape[2]):
                continue
            groups[(r.gpu, r.shape, r.filter)].append(r)
        return {k: min(v, key=lambda r: r.msec) for k, v in groups.items()}

    cud_win = winners("cudnn")
    tc_win = winners("tc", dispatched=True)
    c16_win = winners("c16")
    bad_lines = {(r.key, r.line) for k in SOURCES for r, _kind, _a, _b in defects[k]}

    print("=== 5. blast radius: defective lines that are actually picked as a winner ===")
    for name, win in (("Tensor-Core path  (tc)", tc_win), ("CUDA-Core FP16 path (c16)", c16_win)):
        hit = [r for k, r in win.items() if k in cud_win and (r.key, r.line) in bad_lines]
        n = sum(1 for k in win if k in cud_win)
        print(f"  {name}: {len(hit)} of {n} winning rows sit on a defective line")
        for r in sorted(hit, key=lambda r: (r.gpu, r.shape, r.fw))[:40]:
            want = label_size(r.shape, r.fh, r.fw)
            kinds = ",".join(k for rr, k, _a, _b in defects[r.key] if rr.line == r.line)
            print(f"      {SOURCES[r.key]}:{r.line}  {r.gpu} [{','.join(map(str, r.shape))}] "
                  f"F={r.fw}  Size {r.size} vs label {want:.2f} (x{r.size/want:.3f})  [{kinds}]")
    print()

    print("=== 6. CUDA-Core FP16 speedup distribution, recomputed several ways ===")
    overlap = [k for k in c16_win if k in cud_win]

    def ratios(mode):
        """mode: recorded | fix (relabel) | drop (discard defective) | fix-4090 | rec-4090"""
        out = []
        for k in overlap:
            r = c16_win[k]
            if mode.endswith("-4090") and r.gpu == B.SUSPECT_GPU:
                continue
            if mode.startswith("drop") and (r.key, r.line) in bad_lines:
                continue
            gf = (label_perf(label_size(r.shape, r.fh, r.fw), r.msec)
                  if mode.startswith("fix") else r.gf)
            out.append(gf / cud_win[k].gf)
        return out

    for label, mode in (("as recorded", "recorded"),
                        ("labels fixed", "fix"),
                        ("defective dropped", "drop"),
                        ("fixed, no RTX 4090", "fix-4090"),
                        ("recorded, no 4090", "recorded-4090")):
        summarise(label, ratios(mode), "5.09x mean")
    print()
    summarise("Tensor-Core path", [tc_win[k].gf / cud_win[k].gf for k in tc_win
                                   if k in cud_win], "9.01x mean")
    summarise("TC, no RTX 4090", [tc_win[k].gf / cud_win[k].gf for k in tc_win
                                  if k in cud_win and tc_win[k].gpu != B.SUSPECT_GPU],
              "9.01x mean")
    print("  the tc header's only defects are printed-Size typos and a ratio uses only")
    print("  Performance, so correcting labels cannot move the Tensor-Core statistic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
