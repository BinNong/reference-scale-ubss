#!/usr/bin/env python3
"""方框图几何自检的负向测试：确认 _check_no_overlap 真能抓到「箭头穿框」。

背景：Fig. 2 曾有一条竖箭头的起止 y 恰好等于 red 框的上下边，整条穿框而过，
而渲染/编译都不报错。凡是"自检"，都必须先证明它会失败，否则等于没有。
"""
import sys

sys.path.insert(0, ".")
import make_figures_nfr as M   # noqa: E402

yc = 1.62
box = (3.14, yc - 0.66, 1.72, 0.52)          # "calibrated multiple" 红框
print(f"被检方框 rect={box}  ->  y ∈ [{box[1]}, {box[1] + box[3]}]")

print("\n[1] 负向测试：原来的穿框箭头 (4.00, yc-0.14) -> (4.00, yc-0.66)")
try:
    M._check_no_overlap([box], [(4.00, yc - 0.14, 4.00, yc - 0.66)], "负向测试")
    print("   ✗ 自检没抓到 —— 判据是空的！")
    sys.exit(1)
except SystemExit as e:
    if isinstance(e.code, int) and e.code == 1:
        raise
    print("   ✓ 已抓到并中止：", e)

print("\n[2] 正常测试：图里真实的相邻箭头（红框右缘 -> self-check 左缘）")
try:
    M._check_no_overlap([box], [(4.86, yc - 0.40, 5.08, yc - 0.10)], "正常箭头")
    print("   ✓ 未误报")
except SystemExit as e:
    print("   ✗ 误报了：", e)
    sys.exit(1)

print("\n[3] 全量测试：用 fig_architecture 里真实的 9 个方框 + 9 条箭头")
boxes = [
    (0.05, yc - 0.28, 1.15, 0.56), (1.42, yc + 0.06, 1.5, 0.5),
    (1.42, yc - 0.66, 1.5, 0.52), (3.14, yc + 0.06, 1.72, 0.5),
    (3.14, yc - 0.66, 1.72, 0.52), (5.08, yc - 0.28, 1.35, 0.56),
    (6.65, yc + 0.06, 1.5, 0.5), (6.65, yc - 0.66, 1.5, 0.52),
    (8.45, yc - 0.28, 1.5, 0.56),
]
arrows = [
    (1.20, yc, 1.42, yc + 0.31), (1.20, yc, 1.42, yc - 0.40),
    (2.92, yc + 0.31, 3.14, yc + 0.31), (4.86, yc + 0.31, 5.08, yc + 0.10),
    (4.86, yc - 0.40, 5.08, yc - 0.10), (6.43, yc + 0.10, 6.65, yc + 0.31),
    (6.43, yc - 0.10, 6.65, yc - 0.40), (8.15, yc + 0.31, 8.45, yc + 0.05),
    (8.15, yc - 0.40, 8.45, yc - 0.05),
]
M._check_no_overlap(boxes, arrows, "全量")
print(f"   ✓ {len(arrows)} 条箭头全部未穿框")

print("\n[4] 再负向一次：把那条坏箭头混进全量列表")
try:
    M._check_no_overlap(boxes, arrows + [(4.00, yc - 0.14, 4.00, yc - 0.66)], "混入坏箭头")
    print("   ✗ 没抓到！")
    sys.exit(1)
except SystemExit as e:
    if isinstance(e.code, int) and e.code == 1:
        raise
    print("   ✓ 已抓到：", e)
print("\n几何自检：负向与正向测试全部通过（判据非空）")
