"""校验手稿里所有 Markdown 表格的"均值行"是否等于该列实际平均值。

动机：Table 17 的 "base + energy" 列行值平均为 1.154，表里却写 1.11。
这类错误靠肉眼很难发现，但一查便知。

用法: python3 check_tables.py ../paper/manuscript.md
"""
from __future__ import annotations

import argparse
import re
import sys

NUM = re.compile(r"^-?\$?-?\d+(?:\.\d+)?\$?%?$")


def tofloat(s: str):
    s = s.strip().replace("**", "").replace("$", "").replace(",", "")
    s = s.replace("\\", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def parse_tables(lines):
    tables, cur = [], []
    for ln in lines:
        if ln.strip().startswith("|"):
            cur.append(ln)
        else:
            if len(cur) >= 3:
                tables.append(cur)
            cur = []
    if len(cur) >= 3:
        tables.append(cur)
    return tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    a = ap.parse_args()
    lines = open(a.path, encoding="utf-8").read().split("\n")
    bad = 0
    for ti, tbl in enumerate(parse_tables(lines), 1):
        rows = []
        for ln in tbl:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append(cells)
        if len(rows) < 3:
            continue
        ncol = max(len(r) for r in rows)
        # 判据必须"认标签是不是纯均值标记"，而不是"标签里含 mean 字样"——
        # 后者会把 "mean angle error under the automatic decision" 这类**数据行**误判为均值行（实测踩过）。
        def _is_mean_row(label: str) -> bool:
            t = re.sub(r"[*_$`]", "", label).strip().strip(".。:").strip().lower()
            if t in ("mean", "average", "avg", "均值", "平均"):
                return True
            # 兼容 "mean over 6 regimes" / "mean (dB)" 这类短标签，但要求词数很少且不含数字
            return (re.search(r"\b(mean|average)\b", t) is not None
                    and len(t.split()) <= 3 and not re.search(r"\d", t))

        mean_rows = [r for r in rows if _is_mean_row(r[0])]
        if not mean_rows:
            continue
        data_rows = [r for r in rows if r not in mean_rows]
        for mr in mean_rows:
            for j in range(1, min(len(mr), ncol)):
                stated = tofloat(mr[j])
                if stated is None:
                    continue
                vals = [tofloat(r[j]) for r in data_rows if j < len(r)]
                vals = [v for v in vals if v is not None]
                if len(vals) < 3:
                    continue
                got = sum(vals) / len(vals)
                if abs(got - stated) > 0.012:
                    bad += 1
                    print(f"表#{ti} 第 {j+1} 列: 表内写 {stated:g}，列实际均值 {got:.4f} "
                          f"(n={len(vals)}, 差 {got-stated:+.3f})")
                    print(f"   均值行: |{'|'.join(mr)}|")
    print("\n不一致处数:", bad)

    # ---- 提示性扫描：caption 里的数词 vs 表体规模 ------------------------------
    # 这一项**不判失败**（"five seeds" 之类的数词与表体无关，无法自动判定），
    # 只把两者并排列出来供人扫一眼。真实案例：Table 23 加了第五类噪声后，
    # caption 仍写 "four noise laws"，表体已是五行 —— 这类"文字与表格不同步"
    # 是审稿人最容易抓、脚本最该先筛的一类。
    WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
             9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 15: "fifteen", 20: "twenty"}
    lines = lines.split("\n") if isinstance(lines, str) else lines
    adv = []
    i = 0
    ti = 0
    while i < len(lines):
        m = re.match(r"^\*\*Table (\d+)\.\*\*(.*)$", lines[i])
        if not m:
            i += 1
            continue
        ti += 1
        num, cap = int(m.group(1)), m.group(2)
        j, body, seen_header = i + 1, 0, False
        while j < len(lines) and (lines[j].strip() == "" or lines[j].startswith("|")):
            if lines[j].startswith("|") and not all(
                    set(c) <= set("-: ") for c in lines[j].strip().strip("|").split("|")):
                # 表体的第一行是表头，不是数据行——把它算进去会让
                # Table 23（5 类噪声）报成「数据行 6」，是**误导性提示**。
                if seen_header:
                    body += 1
                seen_header = True
            j += 1
        words = [(w, v) for v, w in WORDS.items() if re.search(rf"\b{w}\b", cap, re.I)]
        if words:
            adv.append((num, body, words, cap[:100]))
        i = j
    if adv:
        print("\n提示（不判失败）：caption 含数词，请对照表体数据行数人工确认")
        for num, body, words, cap in adv:
            print(f"   Table {num:<3} 数据行 {body:<3} 数词 {words}")
            print(f"          {cap}")
    sys.exit(0)


if __name__ == "__main__":
    main()
