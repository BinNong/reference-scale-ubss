# ⚠ 历史一次性 applier，**已被后续轮次取代**，不要对着现行手稿重跑：
#   它写入的是某一轮当时的摘要变体，重跑会回退现行摘要。保留仅为留存记录。
"""把摘要压到 250 词以内（幂等：已是新版则不动）。

用法: python3 compress_abstract.py ../paper/manuscript.md
"""
from __future__ import annotations

import sys

NEW = (
    "Two-stage underdetermined blind source separation (UBSS) clusters the directions of "
    "time–frequency (TF) observations that pass a *single-source point* test, which contains an "
    "energy gate discarding noise-only TF points. We show that the *reference scale* conventionally "
    "used to set that gate — a fixed multiple of the median or of the maximum of the observed "
    "energies — is not invariant to sparsity, and that this misalignment, not the clustering step, "
    "explains much of the paradigm's reported brittleness. The exact mixture distribution of the "
    "per-point energy gives $r(p)=\\mathrm{median}(e)/\\nu$, with $\\nu$ the noise floor: it stays "
    "near unity while the noise-only fraction $\\pi_0=(1-p)^N\\ge1/2$ and grows beyond the *knee* "
    "$p^\\star=1-2^{-1/N}$, so a fixed median-relative threshold drifts by two orders of magnitude "
    "relative to the noise floor. We reference the gate to the noise floor instead, estimate it "
    "blindly from the noise-dominated lower tail through the chi-square quantile relation of the $\\chi^2_{2M}$ law, and "
    "calibrate the threshold to a *false-admission rate*, so that one constant serves the whole "
    "range. A retention self-check disables the gate when its premise fails, degrading to the "
    "classical pipeline rather than failing silently. On a controlled grid the gate attains "
    "$0.30$–$0.53°$ in the sparse regimes against $4.7$–$15.4°$, within a factor of $1.6$ of a "
    "per-regime oracle; on real speech (LibriSpeech, fifteen configurations) it cuts the mean angle "
    "error from $8.26°$ to $1.12°$. Applied unchanged to $k$-means and fuzzy $c$-means front ends "
    "it cuts the angle error by $76$–$97\\%$ and $50$–$90\\%$ at no runtime cost."
)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "../paper/manuscript.md"
    s = open(path, encoding="utf-8").read()
    lines = s.split("\n")
    i = lines.index("## Abstract")
    j = next(k for k in range(i + 1, len(lines)) if lines[k].strip().startswith("**Keywords"))

    n = len(NEW.split())
    print(f"新摘要 {n} 词，$ 个数 {NEW.count('$')}（需为偶数）")
    if n > 250:
        sys.exit(f"仍超 250 词（{n}），不写入")
    if NEW.count("$") % 2:
        sys.exit("$ 未配对，不写入")

    if lines[i + 2].strip() == NEW.strip():
        print("已是新版，未改动")
        return
    # 连空行和旧摘要整段替换（曾误写成 lines[i+1:i+2]，导致新旧两段并存）
    lines[i + 1:i + 3] = ["", NEW]
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    print("已写入")


if __name__ == "__main__":
    main()
