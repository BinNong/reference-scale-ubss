"""R14c：§8.6 与 §8.8 也搬进附录（正文再减约 1.4 页）。

前一回合的提案说这两节"只改两处正文引用"——**对 §8.6 成立，对 §8.8 不成立**：
§8.8 挂着 Fig. 7，而 `build_pdf.insert_figures()` 是在图**首次被引用的那段之后**插图，
搬走后 Fig. 7 会落到 Figs. 8–11 之后，成品校验的"图号落页顺序"必然报错。
所以本轮必须同时做一次图号重排：7→11、8→7、9→8、10→9、11→10
（与第七轮 apply_r7b_fig_renumber 同一类操作）。

改动：
  1. §8.6 → **A.26**（Table 12 → Table 46）；§8.8 → **A.27**；
  2. 图号按上面的映射**单遍**重排（正文引用 + "## Figures" 题注一起改），
     题注列表随后按新号重排顺序，否则清单会是 1..6,11,7,8,9,10；
  3. 六处 "Section 8.6 / 8.8" 引用改为 A.26 / A.27（§1 贡献 2、§6、§7、Table 5、
     以及 Fig. 7 题注自己的 "(Section 8.8)"）；
  4. §8.7 末尾补一句指向 A.26–A.27 的路标，免得正文 §8 从 8.5 跳到 8.7 显得缺节；
  5. 附录导言补一句 A.26–A.27 的来源。

表号不受影响：附录末表现在是 45，新增两张接 46、47。
用法：
    python3 src/apply_r14c_appendix_results.py --dry
    python3 src/apply_r14c_appendix_results.py
"""

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")
TAG = "R14c"

A86_START = "### 8.6 Sensitivity to the false-admission rate"
A86_END = "### 8.6 Sources with genuine time–frequency structure"
A88_START = "### 8.8 How the error scales with the number of TF points"
A88_END = "## 9. Evaluation on Real-Speech Sources"

POINTER_NEW = ("\n\nTwo further studies are reported in the appendix rather than here: "
               "the sensitivity of the rule to the false-admission rate (A.26), and how the "
               "angle error scales with the number of TF points (A.27).")

APP_INTRO_OLD = "moved here so that the main line is not interrupted; their tables are numbered 42–45."
APP_INTRO_NEW = ("and A.26–A.27 the two synthetic studies moved from Sections 8.6 and 8.8; "
                 "all seven were moved out of the main text so that the argument is not interrupted.")
# R15：apply_r15_crossfix 的 A12 又把上句的 "from Sections 8.6 and 8.8" 删掉了
# （§8.8 已不存在、§8.6 已是别的内容），所以"已改"判据要认这句现文本。
APP_INTRO_DONE = ("and A.26–A.27 the two synthetic studies moved out of the main text in an "
                  "earlier round")

# 图号映射：单遍应用，绝不二次替换
FIGMAP = {"7": "11", "8": "7", "9": "8", "10": "9", "11": "10"}
FIG_PAT = re.compile(r"Fig\.\s*(\d+)")

# 表号顺延（与 apply_r14_restructure 同一手法，这里参数化 threshold 与步长）
TAB_LIST = re.compile(r"\bTables?\s+(\d+(?:\s*(?:,|and|–|—|-|to)\s*\d+)*)")
_NUM = re.compile(r"\d+")


def _shift_table_numbers(s: str, lo: int, d: int) -> tuple[str, int]:
    """把每个 'Table(s) …' 里 ≥ lo 的表号加 d。单遍扫描，不会二次替换。"""
    hits = 0

    def rep(m: "re.Match") -> str:
        nonlocal hits
        head = m.group(0)[: m.start(1) - m.start(0)]

        def sub(n: "re.Match") -> str:
            nonlocal hits
            v = int(n.group(0))
            if v >= lo:
                hits += 1
                return str(v + d)
            return n.group(0)

        return head + _NUM.sub(sub, m.group(1))

    return TAB_LIST.sub(rep, s), hits


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def shift_figs(s: str) -> tuple[str, int]:
    hits = 0

    def rep(m: re.Match) -> str:
        nonlocal hits
        n = m.group(1)
        if n in FIGMAP:
            hits += 1
            return m.group(0).replace(n, FIGMAP[n], 1)
        return m.group(0)

    return FIG_PAT.sub(rep, s), hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    p = MS
    s = p.read_text()
    changed = skipped = missing = 0

    # ---------------------------------------------------------------- 1. 切出两节
    def cut(s: str, start: str, end: str) -> tuple[str, str]:
        i = s.index(start)
        j = s.index(end, i)
        return s[:i], s[i:j]

    if "### A.26 " in s and "### A.27 " in s and A86_START not in s:
        skipped += 1
        print(f"  – {TAG} 已落地：A.26/A.27 已在附录，正文无 8.6/8.8")
    else:
        try:
            i86 = s.index(A86_START); j86 = s.index(A86_END, i86)
            i88 = s.index(A88_START); j88 = s.index(A88_END, i88)
        except ValueError as e:
            die(f"找不到 §8.6 / §8.8 的边界：{e}")
        # 1. 图号重排对**全文**做（块还在全文里，块内引用自然一起改）
        s, hits = shift_figs(s)
        if hits == 0:
            die("图号重排命中 0 处——手稿与脚本预期不符")
        # 2. 重新切（shift_figs 只改文本，不动位置），然后把两节从正文摘掉
        i86 = s.index(A86_START); j86 = s.index(A86_END, i86)
        i88 = s.index(A88_START); j88 = s.index(A88_END, i88)
        b86, b88 = s[i86:j86], s[i88:j88]
        s = s[:i88] + s[j88:]          # 先摘后面的，前面各节的位移不受影响
        s = s[:i86] + s[j86:]
        if A86_START in s or A88_START in s:
            die("摘除后正文仍含 8.6/8.8 标题")

        n86 = b86.replace(A86_START, "### A.26 Sensitivity to the false-admission rate", 1)
        n86 = n86.replace("**Table 12.**", "**Table 46.**", 1)
        n86 = n86.replace("established in Section 8.8", "established in A.27", 1)
        n88 = b88.replace(A88_START,
                          "### A.27 How the error scales with the number of TF points", 1)
        n88 = n88.replace("established in Section 8.6", "established in A.26", 1)
        for blk, lab in ((n86, "A.26"), (n88, "A.27")):
            if re.search(r"Section 8\.[68]", blk):
                die(f"{lab} 内仍有 Section 8.6/8.8 引用")
        if "**Table 46.**" not in n86 or "**Table 12.**" in n86:
            die("A.26 的表号未从 12 改到 46")

        # 3. 落点：附录末尾（"## Figures" 之前）
        if s.count("## Figures") != 1:
            die("'## Figures' 不唯一")
        s = s.replace("## Figures", f"\n{n86.rstrip()}\n\n{n88.rstrip()}\n\n## Figures", 1)
        # 4. §8.7 末尾（"## 9." 之前）补路标，免得正文 §8 从 8.5 跳到 8.7 显得缺节
        if s.count(A88_END) != 1:
            die("'## 9. Evaluation …' 不唯一")
        s = s.replace(A88_END, POINTER_NEW.strip() + "\n\n" + A88_END, 1)
        changed += 1
        print(f"  ✓ §8.6 → A.26（Table 46）、§8.8 → A.27；图号重排 {hits} 处；补 §8 路标")

    # ---------------------------------------------------------------- 5. Section 8.x 引用
    # §6 一句里有三种形式：Section 8.6 / Section 8.8 / "Sections 8.1 and 8.8"。
    # 用正则一次改完，逐个列举必漏（第一版就漏了后两种）。
    #
    # R15：触发条件原来按"全文 Section 8.6 / 8.8 的出现次数"。搬移之后正文**合法地**
    # 还有 5 处 "Section 8.6"——它们指向留在正文的结构化声源节（§8.6）。于是每次
    # --dry 都进这个分支、把 5 处合法引用改写成 A.26，只有第 7 步的终检拦着；而终检
    # 报的是"正文仍含 8.6/8.8 小节"，一条永远失败的巡检输出。改成按**标题**判定：
    # 只有 "### 8.6" 标题还在，才说明这两节没搬，才需要改引用。
    has_head = (A86_START in s) or (A88_START in s)
    if has_head:
        n86 = len(re.findall(r"Section 8\.6", s))
        n88 = len(re.findall(r"Section 8\.8", s))
        nlst = len(re.findall(r"Sections 8\.1 and 8\.8", s))
        if not (n86 or n88 or nlst):
            die("正文仍有 8.6/8.8 标题，却找不到对应的 Section 引用")
        s = re.sub(r"Section 8\.6", "A.26", s)
        s = re.sub(r"Section 8\.8", "A.27", s)
        s = re.sub(r"(Sections 8\.1 and )8\.8", r"\1A.27", s)
        changed += 1
        print(f"  ✓ Section 8.6/8.8 引用改号：{n86}+{n88} 处，列表形式 {nlst} 处")
    else:
        skipped += 1
        print("  – 被搬走的两节标题已不在正文（§8.6 现为结构化声源，其 Section 8.6 引用合法），跳过改号")

    # ---------------------------------------------------------------- 6. 图题注重排序
    i = s.index("## Figures")
    j = s.index("## References")
    figs_blk = s[i:j]
    caps = [c for c in figs_blk.split("\n\n") if c.strip().startswith("**Fig.")]
    if len(caps) != 11:
        die(f"图题注不是 11 条：{len(caps)}")
    renum = []
    for c in caps:
        m = re.match(r"\*\*Fig\. (\d+)\.\*\*", c)
        renum.append((int(m.group(1)), c))
    order = [n for n, _ in renum]
    if order == list(range(1, 12)):
        skipped += 1
        print("  – 图题注已按新号排序")
    else:
        renum.sort()
        new_blk = "## Figures\n\n" + "\n\n".join(c for _, c in renum) + "\n\n"
        s = s[:i] + new_blk + s[j:]
        changed += 1
        print(f"  ✓ 图题注按新号重排：{order}")

    # ---------------------------------------------------------------- 7. 表号顺延
    # §8.6 的 Table 12 搬成新表的最后一张，于是 12 又空了：所有 ≥13 的表号 -1
    # （13–45 → 12–44，新表 46 → 45）。与 R14 同一手法，幂等判据是"题注序列已连续"。
    caps_now = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", s)]
    want_pre = list(range(1, 12)) + list(range(13, 47))
    if caps_now == list(range(1, len(caps_now) + 1)):
        skipped += 1
        print(f"  – 表号已连续（1..{max(caps_now)}），跳过 -1 顺延")
    elif caps_now != want_pre:
        missing += 1
        print(f"  ✗ 表号序列既非目标态也非预期前置态：{caps_now}")
    else:
        s2, hits2 = _shift_table_numbers(s, lo=13, d=-1)
        after = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", s2)]
        if hits2 == 0 or after != list(range(1, len(after) + 1)):
            missing += 1
            print(f"  ✗ 顺延后仍不连续：改 {hits2} 处，{after}")
        else:
            s = s2
            changed += 1
            print(f"  ✓ 表号 -1 顺延：{hits2} 处 ≥13 的表号，全文回到 1..{max(after)} 连续")

    # ---------------------------------------------------------------- 8. 附录导言
    # 不写表号——上一步刚顺延过，任何写死的号都会过期（R14 已经栽过一次）。
    if APP_INTRO_OLD in s:
        s = s.replace(APP_INTRO_OLD, APP_INTRO_NEW, 1)
        changed += 1
        print("  ✓ 附录导言补一句 A.26–A.27 的来源")
    elif APP_INTRO_NEW in s or APP_INTRO_DONE in s:
        skipped += 1
        print("  – 附录导言已补 A.26–A.27")
    else:
        missing += 1
        print("  ✗ 未命中：附录导言的 A.26–A.27 来源句（R14/R15 两种措辞都不在）")

    if missing:
        die(f"有 {missing} 条未命中，整体不写入")

    # ------------------------------------------------------------ 终检（写入前也算一遍）
    def final(t: str) -> list[str]:
        body, app = t.split("## Appendix A.")
        bad = []
        if "### A.26 " not in app or "### A.27 " not in app:
            bad.append("A.26/A.27 不在附录")
        # R15：这两条探针原来按裸编号 `### 8.6 ` / `Section 8.6` 查。搬移后 §8.7 被重编号成
        # §8.6（结构化声源），于是正文**合法地**有一个 `### 8.6` 标题和 5 处 `Section 8.6`
        # 引用，终检永远失败。改成按**被搬走的两节的标题**查：标题不在即已搬干净。
        if A86_START in body or A88_START in body:
            bad.append("正文仍含 8.6/8.8 小节（误接纳率敏感性 / FT 标度）")
        # §8.8 确实已不存在；`Section 8.8` 若再出现就是漏改。§8.6 的引用是合法的，
        # 由 check_refs.py 的"节号是否存在"一层负责。
        if "Section 8.8" in t:
            bad.append("仍有 Section 8.8 引用")
        caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", t)]
        secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", t)]
        figs = [int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", t)]
        for lab, seq in (("表", caps), ("附录小节", secs), ("图", figs)):
            if seq != list(range(1, len(seq) + 1)):
                bad.append(f"{lab}编号不连续：{seq}")
        # 图号必须与"首次引用顺序"一致：按正文出现顺序收集
        seen = [int(m.group(1)) for m in re.finditer(r"Fig\.\s*(\d+)", body)]
        firsts = []
        for n in seen:
            if n not in firsts:
                firsts.append(n)
        if firsts != sorted(firsts):
            bad.append(f"图号与首次引用顺序不一致：{firsts}")
        if "Two further studies are reported in the appendix" not in body:
            bad.append("§8 路标句缺失")
        return bad

    bad = final(s)
    if bad:
        die("终检失败：" + "；".join(bad))
    if a.dry:
        print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    p.write_text(s)
    print(f"\n已写入 {p}：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    t = p.read_text()
    print("\n=== 终检 ===")
    for lab, ok in [("A.26/A.27 在附录", "### A.26 " in t and "### A.27 " in t),
                    ("新表（45）在附录", "**Table 45.**" in t),
                    ("无 Section 8.6/8.8 残留", not re.search(r"Section 8\.[68]", t)),
                    ("§8 路标句就位", "Two further studies are reported in the appendix" in t),
                    ("附录导言提 A.26–A.27", "A.26–A.27 the two synthetic studies" in t)]:
        print(f"  {'OK  ' if ok else 'FAIL'} {lab}")
        if not ok:
            die("终检失败")
    body, app = t.split("## Appendix A.")
    seen = [int(m.group(1)) for m in re.finditer(r"Fig\.\s*(\d+)", body)]
    firsts = []
    for n in seen:
        if n not in firsts:
            firsts.append(n)
    print(f"  OK   图号按首次引用顺序：{firsts}")
    print(f"\n✓ {TAG} 全部终检通过（正文 {len(body.split())} 词 / 附录 {len(app.split())} 词）")


if __name__ == "__main__":
    main()
