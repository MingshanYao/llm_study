#!/usr/bin/env python3
"""Static layout checker for SVG figures embedded in HTML. No browser needed.

Estimates bounding boxes from coordinates + text-width heuristics
(CJK/fullwidth char ~= 1.0 x font-size, ASCII ~= 0.55 x font-size) and reports:
  clip-*          text/rect extends past the viewBox (would be visually cut)
  text-overlap    two text bboxes intersect (one likely occludes the other)
  text-crosses-rect  text half-in half-out of a box (label sticking off its bar)
  tiny-font       effective font-size < 9
  dup-id          duplicate id= in the document (markers break across SVGs)
  no-viewbox / parse-error

Usage: check_svg.py PAGE.html [MORE.html ...]
Prints per-svg warnings; exit 0 = clean, 1 = issues found.

Limitations: width estimation is approximate (+-10%); elements inside
transform="" groups and per-tspan multi-line layouts are skipped.
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ASC, DESC = 0.75, 0.25  # ascent/descent as fraction of font-size


def char_w(c, fs):
    o = ord(c)
    if o == 0x20:
        return 0.30 * fs
    if 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF or o >= 0x2E80:
        return fs
    return 0.55 * fs


def text_width(s, fs, bold):
    return sum(char_w(c, fs) for c in s) * (1.06 if bold else 1.0)


def num(v, default=None):
    if v is None:
        return default
    try:
        return float(re.sub(r"px$", "", str(v).strip()))
    except ValueError:
        return default


def style_lookup(el, name):
    st = el.get("style")
    if st:
        for part in st.split(";"):
            k, _, v = part.partition(":")
            if k.strip() == name:
                return v.strip()
    return el.get(name)


def inherited(el, chain, name):
    for e in reversed(chain + [el]):
        v = style_lookup(e, name)
        if v is not None:
            return v
    return None


def intersects(a, b, tol=1.0):
    return not (a[2] <= b[0] + tol or b[2] <= a[0] + tol
                or a[3] <= b[1] + tol or b[3] <= a[1] + tol)


def contains(outer, inner, tol=0.5):
    return (outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1]
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def local(el):
    # strip '{http://www.w3.org/2000/svg}' prefix when the svg has xmlns
    return el.tag.rsplit("}", 1)[-1] if isinstance(el.tag, str) else ""


def walk(el, chain, transformed, texts, rects):
    if el.get("transform"):
        transformed = True
    fs = num(inherited(el, chain, "font-size"), 16.0)
    if local(el) == "text" and not transformed:
        tspans = el.findall("tspan")
        if any(t.get("x") or t.get("y") for t in tspans):
            return  # per-tspan positioned multi-line: skip, bbox unknown
        x = num(el.get("x"))
        y = num(el.get("y"))
        if x is None or y is None:
            return
        x += num(el.get("dx"), 0.0)
        y += num(el.get("dy"), 0.0)
        content = "".join(el.itertext()).strip()
        anchor = inherited(el, chain, "text-anchor") or "start"
        bold = (inherited(el, chain, "font-weight") or "") in \
            ("bold", "bolder", "600", "700", "800", "900")
        w = text_width(content, fs, bold)
        if anchor == "middle":
            x0, x1 = x - w / 2, x + w / 2
        elif anchor == "end":
            x0, x1 = x - w, x
        else:
            x0, x1 = x, x + w
        texts.append({"bbox": (x0, y - ASC * fs, x1, y + DESC * fs),
                      "content": content, "fs": fs, "anchor": anchor,
                      "xy": (round(x, 1), round(y, 1))})
    elif local(el) == "rect" and not transformed:
        x = num(el.get("x"), 0.0)
        y = num(el.get("y"), 0.0)
        w = num(el.get("width"))
        h = num(el.get("height"))
        if None not in (x, y, w, h) and w > 0 and h > 0:
            rects.append({"bbox": (x, y, x + w, y + h),
                          "fill": style_lookup(el, "fill") or ""})
    for ch in el:
        walk(ch, chain + [el], transformed, texts, rects)


def snippet(s, n=14):
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def check_svg(idx, block, issues):
    if "<!DOCTYPE" in block or "<!ENTITY" in block:
        issues.append(("ERROR", "parse-error", "SVG %d contains DOCTYPE/ENTITY; refusing to parse" % idx))
        return
    try:
        root = ET.fromstring(block)
    except ET.ParseError as e:
        issues.append(("ERROR", "parse-error", "SVG %d does not parse as XML: %s" % (idx, e)))
        return
    vb = num(root.get("viewBox"))
    if root.get("viewBox") is None:
        issues.append(("ERROR", "no-viewbox", "SVG %d has no viewBox" % idx))
        return
    parts = [float(p) for p in root.get("viewBox").split()]
    if len(parts) != 4:
        issues.append(("ERROR", "no-viewbox", "SVG %d viewBox is not 'minx miny W H'" % idx))
        return
    W, H = parts[2], parts[3]
    texts, rects = [], []
    walk(root, [], False, texts, rects)

    for t in texts:
        x0, y0, x1, y1 = t["bbox"]
        out = []
        if x0 < -0.5:
            out.append("left by %.0fpx" % -x0)
        if x1 > W + 0.5:
            out.append("right by %.0fpx" % (x1 - W))
        if y0 < -0.5:
            out.append("top by %.0fpx" % -y0)
        if y1 > H + 0.5:
            out.append("bottom by %.0fpx (baseline y=%.0f, viewBox h=%.0f)" % (y1 - H, t["xy"][1], H))
        if out:
            issues.append(("WARN", "clip-viewbox",
                           "text %r at (%s,%s) sticks out %s" %
                           (snippet(t["content"]), t["xy"][0], t["xy"][1], " / ".join(out))))
        if t["fs"] < 9:
            issues.append(("WARN", "tiny-font",
                           "text %r font-size=%.1f (<9)" % (snippet(t["content"]), t["fs"])))

    for r in rects:
        x0, y0, x1, y1 = r["bbox"]
        if x0 < -0.5 or y0 < -0.5 or x1 > W + 0.5 or y1 > H + 0.5:
            issues.append(("WARN", "clip-rect",
                           "rect %s extends past viewBox %dx%d" % ([round(v) for v in r["bbox"]], W, H)))

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            if not a["content"] or not b["content"]:
                continue
            if intersects(a["bbox"], b["bbox"]):
                ov_w = round(min(a["bbox"][2], b["bbox"][2]) - max(a["bbox"][0], b["bbox"][0]), 1)
                issues.append(("WARN", "text-overlap",
                               "%r(%s,%s) overlaps %r(%s,%s) by ~%.0fx%dpx" %
                               (snippet(a["content"]), a["xy"][0], a["xy"][1],
                                snippet(b["content"]), b["xy"][0], b["xy"][1],
                                ov_w, 3)))

    for t in texts:
        if not t["content"]:
            continue
        for r in rects:
            rb = r["bbox"]
            if (rb[2] - rb[0]) < 40 and (rb[3] - rb[1]) < 40:
                continue  # legend chips / bullets: labels next to them are fine
            if (r["fill"] or "").strip().lower() in ("none", "transparent"):
                continue  # decorative outlines/strips: labels may straddle them
            if not intersects(t["bbox"], rb):
                continue
            if contains(rb, t["bbox"]) or contains(t["bbox"], rb):
                continue
            stick = max(rb[0] - t["bbox"][0], t["bbox"][2] - rb[2],
                        rb[1] - t["bbox"][1], t["bbox"][3] - rb[3])
            if stick > 3:
                issues.append(("WARN", "text-crosses-rect",
                               "text %r(%s,%s) crosses box edge %s by %.0fpx" %
                               (snippet(t["content"]), t["xy"][0], t["xy"][1],
                                [round(v) for v in rb], stick)))


def check_file(path):
    html = Path(path).read_text(encoding="utf-8")
    issues = []
    ids = re.findall(r'\bid="([^"]+)"', html)
    seen, dups = set(), []
    for i in ids:
        if i in seen and i not in dups:
            dups.append(i)
        seen.add(i)
    for d in dups:
        issues.append(("ERROR", "dup-id", "duplicate id=%r in document (markers/refs break)" % d))

    blocks = re.findall(r"<svg\b.*?</svg>", html, re.S)
    for idx, block in enumerate(blocks, 1):
        check_svg(idx, block, issues)

    n_err = sum(1 for lvl, _, _ in issues if lvl == "ERROR")
    n_warn = len(issues) - n_err
    if issues:
        for lvl, kind, msg in issues:
            print("  %s [%s] %s" % (lvl, kind, msg))
    print("%s: %d svg blocks, %d errors, %d warnings" % (path, len(blocks), n_err, n_warn))
    return n_err + n_warn


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    total = 0
    for path in sys.argv[1:]:
        total += check_file(path)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
