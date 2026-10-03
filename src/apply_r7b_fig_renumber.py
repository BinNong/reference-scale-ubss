#!/usr/bin/env python3
"""第七轮补遗 B：把图号重排成"按正文首次引用顺序"。

问题（2026-09-19 发现）
======================
成品 PDF 里图**题注的落页顺序**是

    1, 6, 2, 11, 3, 4, 5, 9, 7, 8, 10

不是 1..11。原因是两个约定从第五轮起就不一致：

  * 图号按 `## Figures` 清单的书写顺序给（1..11 = scale / architecture / ...）；
  * 插图位置由 `build_pdf.insert_figures` 决定，规则是"放在**首次引用**该图的
    那一段之后"。

清单顺序 ≠ 首次引用顺序，于是"号"与"印出来的先后"对不上。`build_pdf.py` 头部
早年记下过这个现象（"首次引用顺序非单调，实测 1,6,2,3,4,5,9,7,8"），但当时的
处理是"硬写号绕开 LaTeX 自动编号"——**绕开的是自动编号，不是顺序本身**，所以
缺陷一直印在成品里。

本脚本做的事
============
1. 按首次引用顺序重排图号，使 **号 = 落页顺序 = 1..11**；
2. 重排 `## Figures` 清单的行序，让源稿自身也按号递增（否则清单读起来仍是乱的）；
3. 去掉 §7「Implementation and reproducibility」记录清单里对汇总图的附带提及
   —— 它把这图锚到 §8.1 附近（成品第 20 页），而该图在 §9.5 才被讨论；
4. 同步 r7 答复信与 `verify_r7.py` 里的图号（本轮要交出去的东西）；
5. 同步 applier 脚本里**仍然存活**的字面量（live literal）。

映射（旧 -> 新）与依据
======================
    图文件              旧 -> 新   首次引用处
    fig_scale            1 ->  1   §4.2
    fig_guard            6 ->  2   §5.3（内容在 §5.3，却排在 §5.4 的管线图之后，故原先给 6）
    fig_architecture     2 ->  3   §5.4
    fig_sparsity         3 ->  4   §8.1
    fig_snr              4 ->  5   §8.2
    fig_nsources         5 ->  6   §8.4
    fig_scaling          9 ->  7   §8.8
    fig_real_scale       7 ->  8   §9.2
    fig_real_main        8 ->  9   §9.4
    fig_real_summary    11 -> 10   §9.5
    fig_heatmap         10 -> 11   §A.6

有意不动的东西
==============
* **11 张矢量图文件**：逐张 pdftotext 确认图内不含图号文字（只有 (a)/(b) 面板标
  与图例），所以不需要重生成，也就没有"改了号忘了重画图"的风险。
* **r1..r6 的答复信、`verify_r5.py` / `verify_r6.py` 里的 "Fig. N" 字样**：它们是
  各轮的快照，记的是当时的编号；改了反而失真。残留扫描会把它们列出来供确认。
* **applier 里已成历史的字面量**：同理。只改"当前手稿里仍能找到"的那些，
  于是各 applier 的"未命中"计数在改动前后必须完全一致（判据：不得新增）。

用法：
    python3 src/apply_r7b_fig_renumber.py --dry
    python3 src/apply_r7b_fig_renumber.py
    python3 src/apply_r7b_fig_renumber.py --reverse --dry   # 反向重放，供往返测试

往返测试的用法与边界
====================
`--reverse` 是给**往返测试**用的：把改动后的树复制出来反向重放，再与改动前的备份
逐字节比对（实测 13 个文件全部一致，且反向恢复出的首次引用顺序正是改前的
1,6,2,11,3,4,5,9,7,8,10）。两点要注意：

  * 反向是在 `paper/response_to_review_r7.md` 上**整体换号**的。若在应用之后又往那封信
    里补写含图号的文字（本轮就补了「Figure and table numbering」一节，其中同时写着
    新旧两套号），再反向会把这节也换掉。往返测试的基准状态是**应用当时**的那棵树。
  * `live_literal_spans` 的筛选条件必须包含 `ANCHOR`：正向把 §7 那句里的图号删掉之后，
    该字面量就再没有 `Fig. N` 可匹配；只按图号筛选会让反向看不见它，往返不再逐字节相等
    （实测 apply_r6_fixes.py 的那一条就是这样漏的）。
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"
R7_LETTER = ROOT / "paper" / "response_to_review_r7.md"
VERIFY_R7 = ROOT / "src" / "verify_r7.py"

# 目标顺序：按正文**首次引用**排列，名次即图号
NEW_ORDER = [
    "fig_scale.pdf",          # 1  §4.2
    "fig_guard.pdf",          # 2  §5.3
    "fig_architecture.pdf",   # 3  §5.4
    "fig_sparsity.pdf",       # 4  §8.1
    "fig_snr.pdf",            # 5  §8.2
    "fig_nsources.pdf",       # 6  §8.4
    "fig_scaling.pdf",        # 7  §8.8
    "fig_real_scale.pdf",     # 8  §9.2
    "fig_real_main.pdf",      # 9  §9.4
    "fig_real_summary.pdf",   # 10 §9.5
    "fig_heatmap.pdf",        # 11 §A.6
]
N_FIG = len(NEW_ORDER)

# 改前的编号顺序（2026-09-19 之前的状态）。必须写出来：改成新号以后，旧号无法从
# 文件本身反推，而状态判定（"该正向还是该反向"）恰恰需要它。
OLD_ORDER = [
    "fig_scale.pdf",          # 1
    "fig_architecture.pdf",   # 2
    "fig_sparsity.pdf",       # 3
    "fig_snr.pdf",            # 4
    "fig_nsources.pdf",       # 5
    "fig_guard.pdf",          # 6
    "fig_real_scale.pdf",     # 7
    "fig_real_main.pdf",      # 8
    "fig_scaling.pdf",        # 9
    "fig_heatmap.pdf",        # 10
    "fig_real_summary.pdf",   # 11
]

# 整份文件重排 token 的（非 Python 字面量语境，无歧义）
TOKEN_FILES = [R7_LETTER, VERIFY_R7]

# §7 记录清单里那处把汇总图拖到 §8.1 的附带提及
ANCHOR = "gate-plus-weight design of Section 9.5"
SENT = re.compile(re.escape(ANCHOR) + r" and Fig\. (\d+)\.")

# `Fig. 7a` / `Fig. 11` 都要命中；前导字母限制避免误伤 `eFig.` 之类。
# 第三组是**裸面板号**：稿中有一处 `Fig. 8a and 8b`，第二个 "8b" 前面没有 "Fig."，
# 只换第一处就会留下 `Fig. 9a and 8b` 这种自相矛盾（靠下面的自检兜住）。
TOK = re.compile(r"(?<![A-Za-z])Fig\.\s*(\d+)([a-c]?)((?:(?:\s+and|,)\s*\d+[a-c])*)")
BARE = re.compile(r"\b(\d+)([a-c])\b")


def die(msg: str) -> None:
    print(f"错误：{msg}", file=sys.stderr)
    raise SystemExit(1)


def permute(text: str, m: dict[int, int]) -> tuple[str, int]:
    """把 text 里的 `Fig. N` 按 m 换成 `Fig. m[N]`。re.sub 不会重扫替换结果，安全。"""
    hit = [0]

    def num(n: int) -> int:
        if n not in m:
            die(f"图号 {n} 不在映射里（{sorted(m)}）——先核对稿中图号")
        return m[n]

    def rep(mo: re.Match) -> str:
        hit[0] += 1
        tail = mo.group(3)
        if tail:                                  # `and 8b` / `, 9c` 这类裸面板号
            tail = BARE.sub(lambda t: f"{num(int(t.group(1)))}{t.group(2)}", tail)
        return f"Fig. {num(int(mo.group(1)))}{mo.group(2)}{tail}"

    return TOK.sub(rep, text), hit[0]


def caption_files(text: str) -> dict[int, str]:
    """从 `## Figures` 清单取 号 -> 文件名（文件名是权威，题注尾部必须带该标记）。"""
    blk = text[text.index("## Figures"):text.index("## References")]
    out: dict[int, str] = {}
    for mo in re.finditer(r"^\*\*Fig\. (\d+)\.\*\*(.+)$", blk, re.M):
        ref = re.search(r"\(\s*`([\w./]+\.pdf)`\s*\)\s*\.?\s*$", mo.group(2).strip())
        if not ref:
            die(f"Fig. {mo.group(1)} 的题注尾部没有 (`fig_x.pdf`) 标记")
        out[int(mo.group(1))] = Path(ref.group(1)).name
    if len(out) != N_FIG:
        die(f"图清单 {len(out)} 条，预期 {N_FIG} 条")
    return out


CAP_RE = re.compile(r"^\*\*Fig\. (\d+)\.\*\*")


def reorder_caption_block(text: str) -> str:
    """让清单行按号递增，并**原样保留**首尾结构（标题行、尾部的 `---` 分隔线）。

    直接重建整块会把 `## References` 之前那条 `---` 吃掉——那是源稿的排版分隔，
    与图号无关，不该被本脚本顺手删掉。
    """
    i, j = text.index("## Figures"), text.index("## References")
    lines = text[i:j].split("\n")
    idx = [k for k, ln in enumerate(lines) if CAP_RE.match(ln)]
    if len(idx) != N_FIG:
        die(f"清单块里找到 {len(idx)} 行题注，预期 {N_FIG} 行")
    caps = sorted((lines[k] for k in idx),
                  key=lambda ln: int(CAP_RE.match(ln).group(1)))
    new = ("\n".join(lines[:idx[0]]) + "\n"
           + "\n\n".join(caps) + "\n"
           + "\n".join(lines[idx[-1] + 1:]))
    return text[:i] + new + text[j:]


def edit_sentence(text: str, direction: str, old_no: int, new_no: int) -> tuple[str, int]:
    """§7 那处附带提及。正向删除（用旧号锚定），反向插回（用新号锚定）。

    两个方向都在**换号之前**做，因此锚定用的号是各自起始状态里真实存在的那个。
    不含该句的文本（多数 applier 字面量）返回原样、计数 0。
    """
    if ANCHOR not in text:
        return text, 0
    if direction == "fwd":
        pat = re.compile(re.escape(ANCHOR) + r" and Fig\. %d\." % old_no)
        if pat.search(text):
            return pat.sub(ANCHOR + ".", text), 1
        return text, 0                          # 已经删过
    if re.search(re.escape(ANCHOR) + r" and Fig\.", text):
        return text, 0                          # 已经插回
    # 注意：这里同时出现"模式"与"替换文本"，替换文本**不能带正则转义**——
    # 写成 r" and Fig\. %d." 会把反斜杠写进手稿（实测反向重放产出过 `Fig\. 10`）。
    pat = re.compile(re.escape(ANCHOR) + r"\.")
    return pat.sub(ANCHOR + " and Fig. %d." % new_no, text, count=1), 1


# --------------------------------------------------------------------- live literal
def _line_starts(src: str) -> list[int]:
    starts, off = [], 0
    for ln in src.splitlines(keepends=True):
        starts.append(off)
        off += len(ln)
    starts.append(off)
    return starts


def live_literal_spans(src: str, flat_ms: str) -> list[tuple[int, int]]:
    """找出"内容在**当前手稿**里仍能逐字找到"的字符串字面量的字符区间。

    用 ast.literal_eval 取字面量的真实值（正确处理 `\\%` 这类转义），再折叠空白后
    与手稿比对——直接拿源码文本比会把含转义的存活字面量误判为历史字面量。
    """
    starts = _line_starts(src)
    spans = []
    for tk in tokenize.generate_tokens(io.StringIO(src).readline):
        # 条件要包含 ANCHOR：正向删掉那句里的图号之后，该字面量就再没有 `Fig. N`
        # 可匹配了；若只按图号筛选，反向重放会看不见它，往返就不再逐字节相等
        # （实测 apply_r6_fixes.py 的那一条就是这样漏掉的）。
        if tk.type != tokenize.STRING or not (
                re.search(r"Fig\.\s*\d", tk.string) or ANCHOR in tk.string):
            continue
        try:
            val = ast.literal_eval(tk.string)
        except Exception:
            continue
        if not isinstance(val, str):
            continue
        if re.sub(r"\s+", " ", val).strip() in flat_ms:
            spans.append((starts[tk.start[0] - 1] + tk.start[1],
                          starts[tk.end[0] - 1] + tk.end[1]))
    return spans


def rewrite_spans(src: str, spans: list[tuple[int, int]], fn) -> str:
    """对每个区间做替换；从后往前拼，避免位移。"""
    for s, e in sorted(spans, reverse=True):
        src = src[:s] + fn(src[s:e]) + src[e:]
    return src


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只报告，不写盘")
    ap.add_argument("--reverse", action="store_true", help="反向重放（供往返测试）")
    a = ap.parse_args()

    ms = MS.read_text()
    cur = caption_files(ms)                       # 号 -> 文件（当前状态）
    if set(cur.values()) != set(NEW_ORDER):
        die(f"稿中图文件与预期不符：{sorted(cur.values())}")

    # 状态判定必须写死"改前顺序"，因为一旦改成新号，旧号就没法从文件本身反推了。
    seq = [cur[n] for n in sorted(cur)]
    if seq == NEW_ORDER:
        state = "new"
    elif seq == OLD_ORDER:
        state = "old"
    else:
        die(f"稿中图清单既不是改前顺序也不是改后顺序：{seq}")

    want = "new" if a.reverse else "old"
    if state != want:
        print(f"当前已是「{'改后' if state == 'new' else '改前'}」顺序，"
              f"本次{'反向' if a.reverse else '正向'}无需处理，跳过。")
        return

    perm = {i + 1: NEW_ORDER.index(fn) + 1 for i, fn in enumerate(OLD_ORDER)}
    direction = "rev" if a.reverse else "fwd"
    m = {v: k for k, v in perm.items()} if a.reverse else perm
    # §7 那句的增删在换号之前做，锚定用的号必须取"起始状态里真实存在的那个"：
    # 正向从旧稿出发 -> 用旧号；反向从新稿出发 -> 用新号。
    SUM_FIG = "fig_real_summary.pdf"
    old_no = OLD_ORDER.index(SUM_FIG) + 1
    new_no = NEW_ORDER.index(SUM_FIG) + 1

    print(f"图号重排（{'反向' if a.reverse else '正向'}）：")
    for fn in NEW_ORDER:
        o, n = OLD_ORDER.index(fn) + 1, NEW_ORDER.index(fn) + 1
        mark = "  " if o == n else "->"
        print(f"    {fn:<26} {o:>2} {mark} {n:>2}")
    print(f"    映射 {m}")
    print()

    flat_ms = re.sub(r"\s+", " ", ms)

    # 1) 手稿：先做句子增删，再换号，最后重排清单
    body, n_edit = edit_sentence(ms, direction, old_no, new_no)
    if n_edit != 1:
        die(f"§7 那句没被处理（edit 命中 {n_edit} 次）——核对 {ANCHOR!r} 的现行形态")
    body, n_tok = permute(body, m)
    body = reorder_caption_block(body)
    print(f"  paper/manuscript.md            图号 token {n_tok} 处，§7 附带引用 {'插回' if a.reverse else '删除'} 1 处")

    writes: list[tuple[Path, str]] = [(MS, body)]

    # 2) 整份文件换号
    for p in TOKEN_FILES:
        s = p.read_text()
        s2, k = permute(s, m)
        if k:
            print(f"  {p.relative_to(ROOT)}           图号 token {k} 处")
        writes.append((p, s2))

    # 3) applier 里存活着的字面量
    for p in sorted((ROOT / "src").glob("apply_*.py")):
        s = p.read_text()
        spans = live_literal_spans(s, flat_ms)
        if not spans:
            continue

        def fn(chunk: str) -> str:
            c, _ = edit_sentence(chunk, direction, old_no, new_no)
            c, _ = permute(c, m)
            return c

        s2 = rewrite_spans(s, spans, fn)
        if s2 != s:
            print(f"  {str(p.relative_to(ROOT)):<32}存活字面量 {len(spans)} 处")
        writes.append((p, s2))

    if a.dry:
        print("\n--dry：未写盘。")
        return

    for p, s in writes:
        p.write_text(s)
    print(f"\n已写入 {len(writes)} 个文件。")

    # 4) 改完立刻自检
    after = MS.read_text()
    got = caption_files(after)
    seq = [got[n] for n in sorted(got)]
    expect = OLD_ORDER if a.reverse else NEW_ORDER
    if seq != expect:
        die(f"清单顺序不对：{seq}")
    body_only = after[after.index("## 1. Introduction"):after.index("## Figures")]
    first = {}
    for n in sorted(got):
        mo = re.search(rf"Fig\.\s*{n}(?![0-9])", body_only)
        if not mo:
            die(f"Fig. {n} 在正文里找不到引用，build_pdf 会拒绝插图")
        first[n] = mo.start()
    order = [n for n in sorted(first, key=lambda k: first[k])]

    # 裸面板号自检：每个 `\d+[a-c]` 都必须由同一处 `Fig. N` 的 token 覆盖，
    # 且**面板号必须等于该图的号**——否则就是 `Fig. 9a and 8b` 那种"只换了一半"。
    # 注意单个面板（`Fig. 7a`）的尾巴本来就是空的，不能要求尾巴非空。
    stray = []
    for mo in BARE.finditer(body_only):
        head = body_only[max(0, mo.start() - 40):mo.end()]
        k = head.rfind("Fig. ")
        mm = TOK.match(head, k) if k >= 0 else None
        if not (mm and mm.end() == len(head) and mm.group(1) == mo.group(1)):
            stray.append(head[-46:])
    if stray:
        die(f"有裸面板号没被覆盖：{stray}")

    if a.reverse:
        print(f"自检通过（反向）：清单 {seq[:3]}...，正文首次引用顺序 {order}")
        return
    if order != list(range(1, N_FIG + 1)):
        die(f"正文首次引用顺序仍非 1..11：{order}")
    print(f"自检通过：清单按号递增；正文首次引用顺序 = {order}；"
          f"成品落页顺序将与之相同。")


if __name__ == "__main__":
    main()
