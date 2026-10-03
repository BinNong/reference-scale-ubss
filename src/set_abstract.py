# ⚠ 历史一次性 applier，**已被后续轮次取代**，不要对着现行手稿重跑：
#   它写入的是某一轮当时的摘要变体，重跑会回退现行摘要。保留仅为留存记录。
"""把摘要设为**定稿文本**（第三轮）。

为什么单独一个脚本：摘要整段替换比"逐句替换"更容易出错（第一轮就踩过"新旧两段并存、
而词数检查显示正常"）。这里把定稿文本落在代码里，附带三条断言——
词数上限、必需短语、段落数——这样"摘要应该长什么样"是可复跑、可检验的，
而不是只存在于某次对话里。

用法：
    python3 src/set_abstract.py            # 应用（幂等）
    python3 src/set_abstract.py --dry      # 只报告
"""
from __future__ import annotations

import pathlib
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

MAX_WORDS = 252          # 期刊上限 250，留 2 词余量

# 摘要必须保留的关键论断：前两条是方法，后两条是**如实报告的负面结论**——
# 删掉它们会让摘要比正文更强，是最不能出的错。
REQUIRED = [
    "noise-only points",
    "retention self-check",
    "marginally more accurate",
    "order-of-magnitude penalty",
]

ABSTRACT = (
    "Two-stage underdetermined blind source separation (UBSS) clusters the directions of time–frequency "
    "(TF) observations that pass a *single-source point* test, which contains an energy gate discarding "
    "noise-only points. We show that the *reference scale* conventionally used to set that gate — a fixed "
    "multiple of the median or of the maximum energy — is not invariant to source sparsity, and that this "
    "misalignment accounts for a significant, previously under-emphasised part of the paradigm's brittleness. A "
    "mixture-law analysis gives the reference-scale ratio in semi-analytic form and locates the structural boundary "
    "exactly, where the noise-only component stops carrying half of the probability mass; there a "
    "median-referenced coefficient changes meaning by nearly two orders of magnitude. We reference the gate to "
    "the noise floor instead, estimate it blindly from the noise-dominated lower tail through the chi-square quantile "
    "relation, and calibrate the multiple to a false-admission rate, making the threshold *derived* from the "
    "sensor count and one tolerance rather than selected on data. A retention self-check detects when the "
    "premise fails and reverts to the classical criterion rather than failing silently. On a controlled grid it "
    "approaches a per-regime oracle without per-regime knowledge, across the whole sparsity–SNR plane; on "
    "real speech it removes a silent order-of-magnitude penalty. The same experiments demarcate its limits: on "
    "real signals a max-referenced gate with one chosen coefficient is marginally more accurate, and under "
    "impulsive interference the lower-tail model fails. The rule is a gate, not a clustering method, and "
    "transfers unchanged to the two mask-consuming front ends we test."
)


def main() -> None:
    dry = "--dry" in sys.argv
    n = len(ABSTRACT.split())
    assert n <= MAX_WORDS, f"定稿摘要 {n} 词，超过上限 {MAX_WORDS}"
    assert ABSTRACT.count("$") % 2 == 0, "美元符未配对"
    assert ABSTRACT.count("*") % 2 == 0, "强调星号未配对"
    missing = [k for k in REQUIRED if k not in ABSTRACT]
    assert not missing, f"定稿摘要缺关键短语：{missing}"

    lines = MS.read_text().split("\n")
    if "## Abstract" not in lines:
        print("找不到 ## Abstract"); sys.exit(1)
    i = lines.index("## Abstract")
    # 摘要正文固定在 i+2（i+1 是空行）；先用结构断言防呆，再替换整行
    between = [l for l in lines[i + 1:i + 4] if l.strip() and not l.strip().startswith("**Keywords")]
    if len(between) != 1:
        print(f"摘要与 Keywords 之间有 {len(between)} 个非空段落（应为 1）——先修结构再跑本脚本")
        sys.exit(1)
    if lines[i + 2] == ABSTRACT:
        print(f"已是定稿（{n} 词），未改动")
        return
    lines[i + 2] = ABSTRACT
    if not dry:
        MS.write_text("\n".join(lines))
    print(f"摘要已设为定稿：{n} 词" + ("（--dry，未写入）" if dry else ""))


if __name__ == "__main__":
    main()
