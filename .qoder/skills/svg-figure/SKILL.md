---
name: svg-figure
description: 在 HTML 文档中创建或修复 SVG 示意图（数据流、架构、对比图等），核心是 check_svg.py 静态布局自检循环（估算文本包围盒，检出裁剪/重叠/压框/超小字号/重复 id），专治文字遮挡、元素错位。当用户要求画图、加图、改图、新建带图解的 HTML 报告，或抱怨"图不好看/文字被挡/位置不对"时使用。
---

# SVG 示意图：绘制 + 静态自检循环

## 适用场景

- 在 HTML 页面里新建 SVG 示意图（本仓库 `docs/*.html` 的论文解读图尤其适用）
- 修复已有图的问题：文字遮挡/裁剪、相对位置错、间距不美观

## 铁律

**任何 SVG 图在报告完成前，必须通过 `check_svg.py` 且 0 错误 0 警告。** agent 是"盲画"SVG 的：坐标靠心算，静态检查器是唯一机器可验证的防线。检查通过后仍需 `open` 页面让用户在浏览器终审——静态检查覆盖不了美观度。

## 工作流

1. **规划**：列出盒子/轨道/箭头清单和大致尺寸；中文按每字 ≈ 字号 px 估宽（详见 `references/svg-rules.md`，复杂图动笔前先读它）。
2. **画**：写/改 HTML 中的 SVG。
3. **检查**：

```bash
python3 .qoder/skills/svg-figure/scripts/check_svg.py <page.html>            # 可传多个文件
```

4. **修**：逐条处理报告的问题（见下方「检查器能抓什么」），修完重跑，直到 0 错误 0 警告。同一张图最多迭代 3 轮；仍不过就如实告诉用户差在哪，不要谎报完成。
5. **终审**：`open <page.html>`，让用户在浏览器里看渲染效果，并按 `references/svg-rules.md` §5 的清单自查一遍（美观度、箭头落点、配色一致性——这些静态检查不覆盖）。

## 检查器能抓什么

- `clip-viewbox` / `clip-rect`：文本或矩形伸出 viewBox 四边（底部文字被裁最常见）
- `text-overlap`：两个文本包围盒相交（一个大概率挡住另一个）
- `text-crosses-rect`：文本一半在盒子内一半在外（标签骑在框线上）
- `tiny-font`：有效字号 < 9
- `dup-id`：全文档重复 `id=`（marker 串用会让箭头指错方向）
- `no-viewbox` / `parse-error`

估宽是启发式（CJK ≈ 1.0×字号，ASCII ≈ 0.55×字号，±10%）：**报出来的必须修；没报 ≠ 没问题**，间距、对齐、美观仍靠规划阶段的手算和用户终审。

已知不适用：`transform=""` 组内元素、按 tspan 逐字定位的多行文本不检查；`fill="none"` 的装饰框不参与压框检查；<40px 的小色块（图例 chip）旁的标签不算压框。

## 资源

- `scripts/check_svg.py` — 纯 stdlib 静态检查器，无浏览器依赖，逐条输出 `LEVEL [kind] message`，退出码 0 = 干净 / 1 = 有问题
- `references/svg-rules.md` — 排版规则：文本估宽、网格与间距、本仓库调色板、marker id 唯一性、自检清单
