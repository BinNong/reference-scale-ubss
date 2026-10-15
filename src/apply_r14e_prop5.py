"""R14e：补上缺失的 Proposition 5，并给两处无指针的前向引用加位置指针。

由 `check_refs.py`（本轮新建的交叉引用自检）抓到的四项硬伤：
  1. Proposition 编号断档：1,2,3,4,6,7,8,9 —— **5 是空的**；
  2. Proposition 5 被引用 3 次（§1 贡献 2、A.26 的 α 平台解释、Fig. 11 题注的斜率参考线），
     全文却无定义；
  3. §3.3 里 "it is the former that Proposition 6 uses" —— 提前两节用，且没告诉读者在哪；
  4. §4.2 里 "(Proposition 7)" —— 提前一小节用，同样没指针。

修法：
  1. 把 §6 里那段微扰推导的段首斜体标签 `*Scaling argument.*` 升级为加粗命题标签
     `**Proposition 5 (first-order perturbation bound on the direction error).**`
     ——内容一字不改，只是给它编号（§1 的贡献清单本来就这么称呼它，现在名副其实），
     编号断档与悬空引用一并消失；
  2/3. 三处前向引用补位置指针（§3.3 的 Prop 6、§2 与 §4.2 的 Prop 7）。
     带指针的提前引用是正常学术写法（读者知道去哪看），`check_refs.py` 据此放行；
     没指针的才算缺陷。

用法：
    python3 src/apply_r14e_prop5.py --dry
    python3 src/apply_r14e_prop5.py
"""

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")
TAG = "R14e"

SCALING_LABEL = "*Scaling argument.*"
PROP5_LABEL = "**Proposition 5 (first-order perturbation bound on the direction error).**"

P6_OLD = "and it is the former that Proposition 6 uses."
P6_NEW = "and it is the former that Proposition 6 (Section 4.2) uses."
P7_OLD_A = "which is the condition that fails as sparsity rises (Proposition 7)."
P7_NEW_A = "which is the condition that fails as sparsity rises (Proposition 7, Section 4.3)."
# §4.2 里 Table 1 题注后还有一处（A.19  Sweeps 之前那段）：指的是 4.3 的质量边界
P7_OLD_B = "The last column reports the exact mass boundary of Proposition 7 ($\\pi_0\\ge1/2$)"
P7_NEW_B = "The last column reports the exact mass boundary of Proposition 7 (Section 4.3; $\\pi_0\\ge1/2$)"

# --------------------------------------------------------------------------------------
# R20（2026-10-15）把 Proposition 编号改成随文档出现顺序后，本脚本这四处的编号都变了
# （旧 5→9、6→1、7→2）。**上面的 old/new 常量一字不改** —— 它们记录的是 R14e 当时的
# 文本；这里另列"R20 后的当前态"，**只用于判"已应用"**。
#
# 为什么不能就地改 old/new：改了之后，从 R14e 之前的手稿重放本脚本就会写出错误的编号，
# 而这些脚本的价值就在于可重放。同理不能走"登记（_SUPERSEDED）"——效果明明在，
# 登记等于把一条仍然有效的历史改动说成作废。短签名是这里的第三种选择。
PROP9_LABEL_R20 = ("**Proposition 9 (first-order perturbation bound on the "
                   "direction error).**")
P6_NEW_R20 = "and it is the former that Proposition 1 (Section 4.2) uses."
P7_NEW_A_R20 = ("which is the condition that fails as sparsity rises "
                "(Proposition 2, Section 4.3).")
P7_NEW_B_R20 = ("The last column reports the exact mass boundary of Proposition 2 "
                "(Section 4.3; $\\pi_0\\ge1/2$)")


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    p = MS
    s = p.read_text()
    changed = skipped = missing = 0

    # ------------------------------------------------------------ 1. 命题 5
    if PROP5_LABEL in s or PROP9_LABEL_R20 in s:
        skipped += 1
        print("  – Proposition 5 已编号")
    elif SCALING_LABEL in s and s.count(SCALING_LABEL) == 1:
        s = s.replace(SCALING_LABEL, PROP5_LABEL, 1)
        changed += 1
        print("  ✓ §6 的微扰推导段编号为 Proposition 5（内容一字未改）")
    else:
        missing += 1
        print(f"  ✗ 未命中：{SCALING_LABEL!r} 出现 {s.count(SCALING_LABEL)} 次")

    # ------------------------------------------------------------ 2/3. 位置指针
    # 每项给 (old 的所有形态, new 的所有形态)：后者的第二个形态是 R20 重编号后的当前态。
    for olds, news, lab in (
            ((P6_OLD,), (P6_NEW, P6_NEW_R20), "§3.3 的 Proposition 6 引用"),
            ((P7_OLD_A,), (P7_NEW_A, P7_NEW_A_R20), "§2 的 Proposition 7 引用"),
            ((P7_OLD_B,), (P7_NEW_B, P7_NEW_B_R20), "§4.2 的 Proposition 7 引用")):
        if any(n in s for n in news) and not any(o in s for o in olds):
            skipped += 1
            print(f"  – {lab}：已加指针")
        elif olds[0] in s and s.count(olds[0]) == 1:
            s = s.replace(olds[0], news[0], 1)
            changed += 1
            print(f"  ✓ {lab}：补位置指针")
        else:
            missing += 1
            print(f"  ✗ 未命中：{lab}")

    if missing:
        die(f"有 {missing} 条未命中，整体不写入")

    # ------------------------------------------------------------ 终检
    defs = {int(m.group(1)) for m in re.finditer(r"\*\*Proposition (\d+)\s", s)}
    if sorted(defs) != list(range(1, max(defs) + 1)):
        die(f"终检失败：命题编号仍不连续 {sorted(defs)}")
    # 探针同样要认两种编号形态：R20 之前是 5/6/7，之后是 9/1/2。
    for probes, what in (
            ((PROP5_LABEL, PROP9_LABEL_R20), "§6 微扰段的命题标签"),
            (("a first-order perturbation bound on the resulting direction error",),
             "§1 贡献清单里的称呼"),
            (("Davis–Kahan bound gives",), "§6 的 Davis–Kahan 行"),
            (("Proposition 6 (Section 4.2) uses", P6_NEW_R20), "§3.3 的位置指针"),
            (("(Proposition 7, Section 4.3)", P7_NEW_A_R20), "§2 的位置指针"),
            ((r"Proposition 7 (Section 4.3; $\pi_0\ge1/2$)", P7_NEW_B_R20),
             "§4.2 的位置指针"),
    ):
        if not any(p in s for p in probes):
            die(f"终检失败：{what} 的探针消失 {probes[0][:50]!r}")
    print("\n=== 终检 ===")
    print(f"  OK   命题编号连续：{sorted(defs)}")
    print("  OK   微扰推导四行原文在位（Davis–Kahan、c/sqrt(FT) 推导未动）")
    print("  OK   三处前向引用已带位置指针（§2、§3.3、§4.2）")

    if a.dry:
        print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    p.write_text(s)
    print(f"\n已写入 {p}：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    import subprocess
    r = subprocess.run([sys.executable,
                        str(pathlib.Path(__file__).resolve().parent / "check_refs.py")],
                       capture_output=True, text=True)
    print("\n=== check_refs.py ===")
    print(r.stdout.strip()[-600:])
    if r.returncode != 0:
        die("check_refs.py 仍有硬伤")
    print(f"\n✓ {TAG} 全部通过")


if __name__ == "__main__":
    main()
