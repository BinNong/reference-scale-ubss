"""R14：§10 整节迁入附录，正文只留摘要 + 指向句（对应方案 C 的第一步）。

动机（见 2026-10-05 会话）：正文 41 页 / 附录 18 页，而 CSSP 评审的实际负担来自
**正文的必读量**。§10「Robustness of the calibration」2,360 词 + 4 张表躺在正文里，
却是对 §8 结论的复核、不是论证链的一环；第四轮（apply_r4_structure）做过同样的搬迁，
本轮做第二次。搬迁后：正文 §10 只留约 230 词摘要，研究本体成为 **A.21–A.25**，
表号 19–22 顺延为 **46–49**（不触及其他表号——23 之后全部不动）。

改动：
  1. §10 标题保留（§11/§12 的编号因此不变），内容换成 230 词摘要 + 指向 A.21–A.25；
  2. §10.1–§10.5 原样搬入附录末尾（"## Figures" 之前），标题改为 A.21–A.25；
     表题注 19→46 / 20→47 / 21→48 / 22→49，段内自引 Section 10.x → A.2x；
  3. 附录导言补一句 A.21–A.25 的来源；
  4. §A.19 里指向旧号的兩处（"as in Section 10.1"、"Table 20 varies"）改为新号。

幂等：第二次运行必须报"已落地、全跳过、未命中 0"。
用法：
    python3 src/apply_r14_restructure.py --dry
    python3 src/apply_r14_restructure.py
"""

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")
TAG = "R14"

SEC10_NEW = """## 10. Robustness of the calibration

The calibration rests on four assumptions besides the gate itself — the law of the noise, the geometry of the mixing matrix, the order input, and the reliability of the clustering given that order — and this section states in one paragraph each what each costs when it does not hold. The studies themselves are collected as Appendix A.21–A.25 so that the main line is not interrupted.

*The noise law* (A.21). The $\\chi^2_{2M}$ law is exact for circularly symmetric complex Gaussian noise and for nothing else, and the floor estimate does drift under the five further families tested — from $0.38\\times$ for impulsive to $5.2\\times$ for uniform noise. The gate's *decision* nevertheless survives in all six families, because the estimate is used only to place a threshold far below the signal energies and a threshold wrong by a factor of two still sits far below them. The $\\chi^2$ assumption is therefore a *calibration* assumption rather than a *validity* one, and the guard is what marks where that distinction runs out.

*The mixing geometry and the order* (A.22–A.24). The calibrated gate is insensitive to the minimum column separation in the sparse regimes ($0.29$–$0.46°$ over $5$–$45°$), where the conventional rule is not. The order matters more: the gate nearly eliminates over-counting, from $28.3\\%$ to $1.7\\%$, and pays for it by under-counting in $54.2\\%$ of decisions. Since a missing column costs $90°$ in the matching metric while a spurious one costs almost nothing, the order is best taken from the ungated or lightly gated set and the gate applied afterwards; at the correct order the calibrated gate separates by a factor of thirteen ($0.55°$ against $7.19°$).

*A real recorded room* (A.25). On recorded noise the floor estimate is recoverable only per frequency band — to about $7\\%$ in five of the six rooms — and a reverberant recording offers no noise-only null to calibrate a false-admission rate against, so what it requires is a detector for late reverberation rather than a better calibration. That study validates the failure analysis of A.21 on recorded noise and diagnoses the modelling component a reverberant setting would need; it is not an end-to-end validation of the detector on reverberant mixtures.

"""

# 放在 A.21 标题下的导言段（原 §10 导言的改写：它点名五个小节，只能放在第一个小节里）
A21_LEAD = """These five subsections (A.21–A.25) report where the calibration is, and is not, insulated from the assumptions it rests on: the law of the noise (this subsection), the geometry of the mixing matrix (A.22), the order input (A.23), the sensitivity of the clustering to a wrong order (A.24) and — the one study of the five that uses no synthetic component at all — a real recorded room (A.25). Their real-speech counterparts are Sections 9.2 and 9.4; Section 10 in the main text states the one-paragraph summary of each."""

# 搬过来的文本里必须先改的子串（顺序无关，两两不重叠）
RENUMBER = [
    ("### 10.1 Departures from the Gaussian noise law",
     "### A.21 Departures from the Gaussian noise law"),
    ("### 10.2 Sensitivity to the conditioning of the mixing matrix",
     "### A.22 Sensitivity to the conditioning of the mixing matrix"),
    ("### 10.3 The gate's strictness and automatic order selection",
     "### A.23 The gate's strictness and automatic order selection"),
    ("### 10.4 Sensitivity to the assumed source number",
     "### A.24 Sensitivity to the assumed source number"),
    ("### 10.5 A real recorded room", "### A.25 A real recorded room"),
    ("**Table 19.**", "**Table 46.**"),
    ("**Table 20.**", "**Table 47.**"),
    ("**Table 21.**", "**Table 48.**"),
    ("**Table 22.**", "**Table 49.**"),
    ("Table 19 makes", "Table 46 makes"),
    ("Section 10.3", "A.23"),
    ("Section 10.2", "A.22"),
    ("Section 10.1", "A.21"),
]

# 附录导言追加的一句。
# R15：原来这句写着 "their tables are numbered 42–45"。R14c 随后把 ≥13 的表号整体 −1，
# 那四张表现在是 41–44；而 APP_INTRO_OLD 仍然匹配当前导言，于是一次非演练重跑会把
# 这个**已经过时的表号**再插一遍，脚本自己的终检却只回查刚插入的字符串、看不出来。
# 因此这里不再写死表号——本脚本 :243 的注释早已警告过同一个坑。
APP_INTRO_OLD = "Table numbers continue from the\nmain text."
APP_INTRO_NEW = ("Table numbers continue from the\n"
                 "main text. A.21–A.25 are the five robustness studies that Section 10 summarises, "
                 "moved here so that the main line is not interrupted; their table numbers are "
                 "listed in the repository README's record table, which is regenerated from the "
                 "records rather than written down here.")

# R14c 重写后的导言（现在的目标态）。见到它就说明第 2 步已被后续轮次取代。
_R14C_INTRO = ("A.21–A.25 are the five robustness studies that Section 10 summarises, "
               "and A.26–A.27")

# §A.19 里两处指向旧号的引用
A19_OLD_1 = "distinction as in Section 10.1, now on recorded rather than synthetic noise"
A19_NEW_1 = "distinction as in A.21, now on recorded rather than synthetic noise"
A19_OLD_2 = "Table 20 varies the *minimum column separation* and finds the algorithm insensitive to it"
A19_NEW_2 = "Table 47 varies the *minimum column separation* and finds the algorithm insensitive to it"

# ---------------------------------------------------------------- 5. 表号顺延
# §10 的四张表（19–22）搬到附录末尾后先占 46–49，而附录原有表是 23–45 ——
# 于是 19–22 空了出来。把**所有 ≥23 的表号统一减 4**（46–49 → 42–45，23–45 → 19–41），
# 全文回复 1..45 连续。正文末表是 18，正好接上附录的 19。
# 幂等判据不是"再减一次"，而是**目标态**：题注序列已连续就跳过。
TAB_LIST = re.compile(r"\bTables?\s+(\d+(?:\s*(?:,|and|–|—|-|to)\s*\d+)*)")
_NUM = re.compile(r"\d+")


def _shift_table_numbers(s: str) -> tuple[str, int]:
    """把每个 'Table(s) …' 里的表号 ≥23 者减 4。单遍扫描，不会二次替换。"""
    hits = 0

    def rep(m: re.Match) -> str:
        nonlocal hits
        head = m.group(0)[: m.start(1) - m.start(0)]
        body = m.group(1)

        def sub(n: re.Match) -> str:
            nonlocal hits
            v = int(n.group(0))
            if v >= 23:
                hits += 1
                return str(v - 4)
            return n.group(0)

        return head + _NUM.sub(sub, body)

    return TAB_LIST.sub(rep, s), hits


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

    # ---------------------------------------------------------------- 1. 切出 §10
    try:
        i0 = s.index("## 10. Robustness of the calibration")
        i1 = s.index("## 11. Discussion and Limitations")
    except ValueError:
        die("找不到 §10 / §11 标题——手稿结构与脚本预期不符")

    block = s[i0:i1]
    if "### 10.1 Departures" not in block:
        skipped += 1
        print(f"  – {TAG} 已落地：§10 已是摘要节，未发现 10.1–10.5")
    else:
        j0 = block.index("### 10.1 ")
        head, studies = block[:j0], block[j0:]
        # 守卫：head 必须是"标题 + 原导言段"，导言段以 "The studies collected here" 开头
        if not head.startswith("## 10. Robustness of the calibration\n\n"):
            die(f"§10 头部形状异常：{head[:80]!r}")
        if "The studies collected here" not in head:
            die("§10 导言段不是预期的 'The studies collected here …'")
        if "\n### " in studies.rstrip() and studies.count("### ") != 5:
            die(f"搬迁块的小节数不是 5：{studies.count('### ')}")
        # 五个表题注、五处标题与全部自引都必须至少命中一次（"Section 10.3" 在块内出现 3 次）
        for old, new in RENUMBER:
            n = studies.count(old)
            if n < 1:
                die(f"重编号子串 {old!r} 出现 {n} 次，预期至少 1 次")
        new_studies = studies
        for old, new in RENUMBER:
            new_studies = new_studies.replace(old, new)
        if re.search(r"Section 10\.\d|Table (19|20|21|22)\b", new_studies):
            die("重编号后仍有旧号残留")
        # 导言段放进 A.21 标题之下（原 §10 导言的改写，登记而非删除）
        anchor21 = "### A.21 Departures from the Gaussian noise law\n\n"
        if new_studies.count(anchor21) != 1:
            die("A.21 标题锚点不唯一")
        new_studies = new_studies.replace(
            anchor21, anchor21 + A21_LEAD + "\n\n", 1)
        # 落点：附录末尾（"## Figures" 之前）；与现有一致，不加分隔线
        if s.count("## Figures") != 1:
            die("'## Figures' 不唯一，无法定位附录末尾")
        s = (s[:i0] + SEC10_NEW + s[i1:])
        s = s.replace("## Figures", f"\n{new_studies.rstrip()}\n\n## Figures", 1)
        changed += 1
        print(f"  ✓ {TAG} §10 → A.21–A.25（表 19–22 → 46–49），正文换摘要")

    # ---------------------------------------------------------------- 2. 附录导言
    # R15：这一条已被 apply_r14c 取代——R14c 把导言整句重写过（还补了 A.26–A.27 的来源），
    # 现在这句话已经在稿里，只是措辞不同。原实现按 "APP_INTRO_OLD 是否在场" 判定，
    # 而 OLD 是 "Table numbers continue from the main text."，它至今仍是新句子的前缀，
    # 于是每次 --dry 都报"将改 1"，一次非演练重跑会把已过时的表号 42–45 再插一遍。
    # 按仓库惯例登记为 superseded：不删条目（删了就看不见这次改动发生过），但不再触发。
    if _R14C_INTRO in s:
        skipped += 1
        print("  – 附录导言的 A.21–A.25 来源句已由 apply_r14c 重写，本条目 superseded")
    elif APP_INTRO_OLD in s:
        if APP_INTRO_NEW in s:
            skipped += 1
            print("  – 附录导言已补 A.21–A.25 的来源")
        else:
            s = s.replace(APP_INTRO_OLD, APP_INTRO_NEW, 1)
            changed += 1
            print("  ✓ 附录导言补一句 A.21–A.25 的来源")
    else:
        missing += 1
        print("  ✗ 未命中：附录导言 'Table numbers continue from the main text.'")

    # ---------------------------------------------------------------- 3. §A.19 旧号
    # done 探针取"顺延前后都成立"的稳定尾巴，避免把第 5 步的 -4 误判成二次未命中
    for old, new, done, lab in (
            (A19_OLD_1, A19_NEW_1, "distinction as in A.21",
             "A.19 的 Section 10.1 → A.21"),
            (A19_OLD_2, A19_NEW_2,
             "varies the *minimum column separation* and finds the algorithm insensitive to it",
             "A.19 的 Table 20 → 迁移后随 -4 顺延到 Table 43")):
        if old in s:
            s = s.replace(old, new, 1)
            changed += 1
            print(f"  ✓ {lab}")
        elif done in s:
            skipped += 1
            print(f"  – {lab}：已改")
        else:
            missing += 1
            print(f"  ✗ 未命中：{lab}")

    # ---------------------------------------------------------------- 5. 表号顺延
    caps_now = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", s)]
    want_pre = list(range(1, 19)) + list(range(23, 50))     # 1..18 + 23..49
    if caps_now == list(range(1, len(caps_now) + 1)):
        skipped += 1
        print(f"  – 表号已连续（1..{max(caps_now)}），跳过 -4 顺延")
    elif caps_now != want_pre:
        missing += 1
        print(f"  ✗ 表号序列既非目标态也非预期前置态：{caps_now}")
    else:
        s2, hits = _shift_table_numbers(s)
        after = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", s2)]
        if hits == 0 or after != list(range(1, len(after) + 1)):
            missing += 1
            print(f"  ✗ 顺延后仍不连续：改 {hits} 处，{after}")
        else:
            s = s2
            changed += 1
            print(f"  ✓ 表号 -4 顺延：{hits} 处 ≥23 的表号，全文回到 1..{max(after)} 连续")

    if missing:
        die(f"有 {missing} 条未命中，整体不写入")
    if a.dry:
        print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    p.write_text(s)
    print(f"\n已写入 {p}：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    # ------------------------------------------------------------ 终检
    t = p.read_text()
    print("\n=== 终检 ===")
    body = t.split("## Appendix A.")[0]
    app = "## Appendix A." + t.split("## Appendix A.")[1]
    checks = [
        ("正文保留 §10 标题", "## 10. Robustness of the calibration" in body),
        ("正文只剩摘要（无 10.1–10.5）", "### 10.1 " not in body),
        ("附录含 A.21–A.25 五个标题",
         all(f"### A.{n} " in app for n in range(21, 26))),
        ("表 42–45 在附录（原 §10 四表）", all(f"**Table {n}.**" in app for n in range(42, 46))),
        ("正文表格止于 18", not any(f"**Table {n}.**" in body for n in range(19, 46))),
        ("全稿无 'Section 10.x' 残留", not re.search(r"Section 10\.\d", t)),
        ("A.21 导言段就位", A21_LEAD[:60] in app),
        ("附录导言已补来源", "moved here so that the main line is not interrupted" in app),
        ("附录导言表号 42–45 与顺延后一致", "tables are numbered 42–45." in app),
        ("A.19 引 A.21", "distinction as in A.21" in app),
        ("A.19 引 Table 43（原 Table 20，已随 -4 顺延）",
         "Table 43 varies the *minimum column separation*" in app),
    ]
    for lab, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {lab}")
        if not ok:
            die("终检失败")
    caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", t)]
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", t)]
    figs = [int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", t)]
    for lab, seq in (("表", caps), ("附录小节", secs), ("图", figs)):
        ok = seq == list(range(1, len(seq) + 1))
        print(f"  {'OK  ' if ok else 'FAIL'} {lab}：共 {len(seq)}，1..{max(seq)} 连续无重复")
        if not ok:
            die(f"{lab}编号不连续或有重复：{seq}")
    print(f"\n✓ {TAG} 全部终检通过（正文 {sum(1 for _ in body.splitlines())} 行 / "
          f"附录 {sum(1 for _ in app.splitlines())} 行）")


if __name__ == "__main__":
    main()
