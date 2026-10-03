"""R8-3  把本轮新增/扩充的归档写进 §7 的数据可用性清单。

论文声称 "every machine-readable record named above are available at ..."，
所以**每新增一个归档都必须在清单里点名**，否则声明与事实不符（这正是本项目的
"记录必须与手稿同步"那条纪律）。本轮涉及两处：

  1. `r2_noise_family.json` 由 120 条扩到 144 条（新增 Student-t 一族，Table 19 的第六行）；
  2. 新增 `r8_n_mismatch.json`（1,200 条，§10.4 的 Table 22）。

幂等：再跑一次应报"已改 0 | 跳过 2 | 未命中 0"。
用法：
    python3 src/apply_r8c_manifest.py --dry
    python3 src/apply_r8c_manifest.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"

E: list[tuple[str, str, str]] = [
    ("§7 清单：r2_noise_family 记录数 120 -> 144",
     "`r2_noise_family.json` ($120$)",
     "`r2_noise_family.json` ($144$)"),

    ("§7 清单：补第八轮归档 r8_n_mismatch.json",
     "which is the record behind the corrected Table 39 and Appendix A.15.",
     "which is the record behind the corrected Table 39 and Appendix A.15. The eighth round adds "
     "`r8_n_mismatch.json` ($1{,}200$) for the source-number mismatch study of Section 10.4, and extends "
     "`r2_noise_family.json` from $120$ to $144$ records by adding the Student-$t$ family of Table 19."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    s = MS.read_text()
    before = s
    changed = skipped = 0
    for tag, old, new in E:
        if new in s:
            skipped += 1
            print(f"  [跳过] {tag}")
            continue
        if old not in s:
            print(f"✗ 未命中：{tag}")
            sys.exit(1)
        if s.count(old) != 1:
            print(f"✗ 「{tag}」旧文出现 {s.count(old)} 次，要求唯一")
            sys.exit(1)
        s = s.replace(old, new, 1)
        changed += 1
        print(f"  [已改] {tag}")
    print(f"\n已改 {changed} | 跳过 {skipped} | 未命中 0")
    if s == before:
        print("稿件无变化。")
        return
    if s.count("$") % 2:
        print(f"✗ $ 不配对：{s.count('$')}")
        sys.exit(1)
    if a.dry:
        print("[dry] 未写入。")
        return
    MS.write_text(s)
    print(f"已写入 {MS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
