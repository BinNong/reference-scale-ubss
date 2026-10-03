"""R8-4 把真实录音房间研究写进手稿（CSSP 意见 Major 5 / Minor 8）。

新增 **§10.5「A real recorded room」**（正文，叙述结论 + 指向表格）与
**§A.18 + Table 43**（附录，放表格与实验设定）。

**为什么表放附录**：表号按首次出现顺序给，往正文中间插一张表会让附录 23–42 整体再移一次号
（第八轮已经移过一次）。本轮把表追加在附录末尾（Table 43），正文 §10.5 报结论并指路——
这既避免第二次重排，也与全稿"稳健性研究放附录"的既有分工一致。
老师若要把它提到正文，说一声即可（代价是附录表再整体移一号）。

同时把新归档写进 §7 的可用性清单。

幂等：再跑一次应报"已改 0 | 跳过 4 | 未命中 0"。
用法：
    python3 src/apply_r8d_realroom.py --dry
    python3 src/apply_r8d_realroom.py
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"

A18_TITLE = "### A.18 The noise floor of a real recorded room"
A18_TEXT = """
Sections 9 and 10.5 use recorded speech; this section replaces the synthetic noise by a
recording. The material is OpenSLR SLR28, the room-impulse-response and noise database
released with the 2014 REVERB challenge: eight-microphone array impulse responses measured
in six real rooms (two small, two medium, two large) and isotropic noise recorded in the
same rooms with the same array, all at $16$ kHz. Two of the eight channels are used
($M=2$), and the analysis is that of Section 5.1 applied to the noise alone: per-point
energy, lower-tail inversion, and the $\chi^2_{2M}$ law the inversion assumes.

The estimator is applied twice. In the *global* reading it is applied once to the whole
time--frequency plane, which is what Section 9 does and what a user would do. In the
*per-band* reading it is applied separately to each of the $129$ frequency bins, and the
reported statistic is the ratio of the empirical $5\%$ quantile of $2e/\sigma^2$ within a
bin to the $\chi^2_{2M}$ quantile at the same level, summarised across bins.

**Table 43.** The lower-tail model on recorded room noise (six rooms, eight-microphone
REVERB 2014 recordings, two channels used). "global" applies the inversion of Section 5.1
once to the whole plane; "per band" applies it to each of $129$ bins separately. The
global ratio is the empirical $5\%$ quantile of $2e/\\sigma^2$ divided by the
$\\chi^2_{2M}$ quantile at the same level, so $1$ is agreement; the per-band column
reports the median of that ratio over bins together with the $10$--$90\\%$ spread. "band
range" is the ratio of the largest to the smallest bin-mean noise power, in dB.

| Room | global $\\hat\\sigma^2/\\sigma^2$ | band range (dB) | global tail ratio, $q=0.05$ | per-band tail ratio (median, $10$--$90\\%$) |
|---|---|---|---|---|
| small room 1 | $0.0003$ | $59.0$ | $0.0010$ | $0.939$ / $0.828$--$1.013$ |
| small room 2 | $0.0120$ | $38.3$ | $0.0287$ | $0.966$ / $0.891$--$1.017$ |
| medium room 1 | $0.0001$ | $58.5$ | $0.0002$ | $0.474$ / $0.268$--$0.801$ |
| medium room 2 | $0.0014$ | $46.8$ | $0.0044$ | $0.978$ / $0.892$--$1.049$ |
| large room 1 | $0.0001$ | $61.4$ | $0.0002$ | $0.927$ / $0.854$--$0.990$ |
| large room 2 | $0.0140$ | $41.0$ | $0.0279$ | $0.991$ / $0.908$--$1.046$ |

The global reading fails in every room, by a factor of $71$ in the mildest case
(small room 2) and by $1.5$ to $4$ orders of magnitude in the others: the pooled lower
tail is set by the low-frequency bins, whose power exceeds the median bin by up to
$61$ dB, and a single scale cannot describe a distribution that wide. The per-band
reading restores the model in five of the six rooms, the median ratio falling between
$0.927$ and $0.991$ with a spread of about $\\pm7\\%$. The exception is medium room 1,
where it falls to $0.474$: a narrowband component in that recording — building or
air-conditioning tone — is not described by a $\\chi^2$ lower tail at the bins it occupies,
and no scale correction repairs it. This is the same *calibration-versus-validity*
distinction as in Section 10.1, now on recorded rather than synthetic noise, and with the
banded floor of Appendix A.14 as the repair rather than the guard.
"""

S105_TITLE = "### 10.5 A real recorded room"
S105_TEXT = """
Every study in this section so far is synthetic. This one is not. It uses the
eight-microphone array impulse responses and the isotropic-noise recordings that the 2014
REVERB challenge measured in six real rooms and released as OpenSLR SLR28, so that both
the acoustics and the noise are recorded rather than generated; the protocol is
Appendix A.18.

The first question a calibration of this kind invites is whether its lower-tail premise
survives contact with a real recording. It survives it *per frequency band*, and it does
not survive it globally. Pooled over the whole time--frequency plane the floor estimate of
Section 5.1 comes out low by a factor of $71$ in the mildest of the six rooms and by
$1.5$ to $4$ orders of magnitude in the others, because recorded room noise is not white:
its power spans $38$ to $61$ dB across the band, the low frequencies being dominated by
building and air-conditioning noise. Applied band by band instead, the same inversion is
accurate to about $7\\%$ in five of the six rooms, with a median lower-tail ratio between
$0.927$ and $0.991$; in the sixth a narrowband component takes it down to $0.474$
(Table 43). Two consequences follow. The single global floor used in Section 9 is a
convenience and not a property of the method — on a recorded floor the estimate has to be
taken per band, which is the banded variant of Appendix A.14 — and the *validity* of the
$\chi^2$ shape is a separate question from its calibration, one that a single room with a
tonal component is already enough to settle in the negative.

The second question is whether a genuinely reverberant recording offers anything to
calibrate a *false-admission rate* against, and here the answer is no. Convolving the
measured impulse responses with the LibriSpeech sources of Section 9 and adding the
recorded noise of the same room at $20$ dB, a per-band calibrated gate at
$\\alpha=10^{-4}$ admits between $24\\%$ and $83\\%$ of the time--frequency points at which
no source is emitting. That is either a large excess of false admissions or evidence that
those points are not noise-only, and the second reading is the right one: the
reverberation tail of an earlier emission lies above the recorded floor there, so the
class an activity oracle would call "noise" is a mixture of noise and late reverberation.
A false-admission rate on reverberant recordings therefore has to be measured on the
recording's own noise, as Table 43 does, and not against an activity oracle. What the
real room adds to Section 10.2 is not a new transferable criterion but a sharper statement
of what is missing: with a recorded floor the *reference scale* is recoverable per band,
while the *decision* would need a detector that distinguishes late reverberation from
noise — a detection problem, not a calibration one.
"""

E: list[tuple[str, str, str]] = [

("§10 引言：加入 §10.5（实测房间）",
 "The studies collected here report where the calibration is, and is not, insulated from the assumptions it "
 "rests on: the law of the noise (Section 10.1), the geometry of the mixing matrix (Section 10.2), the order "
 "input (Section 10.3) and the sensitivity of the clustering to a wrong order (Section 10.4). All four are "
 "synthetic; their real-speech counterparts are Sections 9.2 and 9.4.",

 "The studies collected here report where the calibration is, and is not, insulated from the assumptions it "
 "rests on: the law of the noise (Section 10.1), the geometry of the mixing matrix (Section 10.2), the order "
 "input (Section 10.3), the sensitivity of the clustering to a wrong order (Section 10.4), and — the one study "
 "of the five that uses no synthetic component at all — a real recorded room (Section 10.5). Their real-speech "
 "counterparts are Sections 9.2 and 9.4."),

("§7 清单：补 r8_real_room.json",
 "by adding the Student-$t$ family of Table 19.",
 "by adding the Student-$t$ family of Table 19. The same round adds `r8_real_room.json` ($54$) for the recorded "
 "room of Section 10.5 and Table 43, which uses the measured impulse responses and recorded isotropic noises of "
 "the 2014 REVERB challenge as released in OpenSLR SLR28."),
]


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    s = MS.read_text()
    # 修：Table 43 的题注必须是**单行**。`table_captions` 把 `**Table N.** cap` 转成
    # pandoc 的 `: cap`，而它在 `: cap` 与续行之间插了一个空行——pandoc 于是把题注
    # 当普通段落渲染成 ": ..."，longtable 拿不到 caption。稿件里既有题注全是超长单行。
    s = re.sub(r"(\*\*Table 43\.\*\*)(.*?)\n\n\| Room",
               lambda m: m.group(1) + m.group(2).replace("\n", " ") + "\n\n| Room",
               s, count=1, flags=re.S)
    # 修掉上一次误写的 BEL：'\alpha' 写在非 raw 串里时 \a 会塌缩成 BEL(0x07)，
    # 于是原文成了 "<BEL>lpha"——要还原成 "\alpha"（BEL -> \a，两字符），
    # 不是 "\lpha"（第一版就修错成这样，LaTeX 报 Undefined control sequence）。
    s = s.replace("\x07", "\\a").replace("\\lpha", "\\alpha")
    before = s
    changed = skipped = 0

    for tag, old, new in E:
        if new in s:
            skipped += 1
            print(f"  [跳过] {tag}")
            continue
        if old not in s or s.count(old) != 1:
            die(f"「{tag}」：旧文出现 {s.count(old)} 次（要求唯一）")
        s = s.replace(old, new, 1)
        changed += 1
        print(f"  [已改] {tag}")

    # ---- §10.5 插在 §11 之前 ----
    if S105_TITLE in s:
        skipped += 1
        print("  [跳过] §10.5")
    else:
        i11 = s.index("## 11. Discussion and Limitations")
        s = s[:i11] + S105_TITLE + "\n" + S105_TEXT + "\n" + s[i11:]
        changed += 1
        print("  [已改] 插入 §10.5")

    # ---- §A.18 + Table 43 追加在附录末尾 ----
    if A18_TITLE in s:
        skipped += 1
        print("  [跳过] §A.18 + Table 43")
    else:
        i_fig = s.index("## Figures")
        s = s[:i_fig] + A18_TITLE + "\n" + A18_TEXT + "\n" + s[i_fig:]
        changed += 1
        print("  [已改] 追加 §A.18 + Table 43")

    # ---- 终检 ----
    secs = [int(x) for x in __import__("re").findall(r"(?m)^### A\.(\d+) ", s)]
    if secs != list(range(1, len(secs) + 1)):        # 只判连续无重复，不写死节数
        die(f"附录小节编号不连续或有重复，实为 {secs}")
    caps = [int(x) for x in __import__("re").findall(r"\*\*Table (\d+)\.\*\*", s)]
    if caps != list(range(1, len(caps) + 1)):        # 只判连续无重复，不写死张数
        die(f"表题注编号不连续或有重复，实为 {caps}")
    i10 = s.index("## 10. Robustness of the calibration")
    i105 = s.index(S105_TITLE)
    i11 = s.index("## 11. Discussion and Limitations")
    if not (i10 < i105 < i11):
        die("§10.5 不在 §10 内")
    if s.count("$") % 2:
        die(f"$ 不配对：{s.count('$')}")
    print(f"\n已改 {changed} | 跳过 {skipped} | 未命中 0")
    print(f"词数 {len(before.split())} -> {len(s.split())}")
    if a.dry:
        print("[dry] 未写入。")
        return
    MS.write_text(s)
    print(f"已写入 {MS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
