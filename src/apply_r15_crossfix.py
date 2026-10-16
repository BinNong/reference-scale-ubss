#!/usr/bin/env python3
"""R15 投稿前审核修订：把 paper/manuscript.md 里审核发现的跨引用/数字/重复段问题一次改掉。

背景：R14 系列四轮搬移（§10 → A.21–A.25、§8.6/§8.8 → A.26/A.27）把 45 张表和 11 张图的编号整体顺延过
三次。check_refs.py 只查「引用是否悬空」「编号是否连续」，查不出「一个语义正确的旧编号指向了内容不同的
新表」；verify_* 只查它们自己那一轮的表。r15_recount.py 补上数字层后，暴露出下面 A 组 12 处硬错误。

用法（从仓库根运行）：

    python3 src/apply_r15_crossfix.py --dry     # 只报不改
    python3 src/apply_r15_crossfix.py           # 落地

惯例（与 apply_r8*/apply_r10*/apply_r14* 一致）：
  * old/new 都是字面文本；new 不得包含整个 old，否则每条都会反复触发；
  * 幂等判据是「无新增未命中」，不是零未命中；
  * 跑完自回调 check_refs.py 作后置条件。
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MS = os.path.join(ROOT, "paper", "manuscript.md")

# ---- 以下五条的 old 由稿件原文用 repr() 生成，避免转义层错配 ----
A1_OLD = '\n\nFirst, the energy weight alone fails at SNR $0$ dB ($7.32°$, against $1.41°$ for the gate): when signal and noise energies are comparable the weight cannot separate them, and the hard removal performed by the gate is still required — it is the *combination* of gate and weight that we recommend. Second, the synthetic ladder gives no gain at all, and the sign there is the one the amplitude–purity relation predicts: the weighted mean is $2.49°$ against $2.12°$ for the plain gate, worse in four of the six regimes, with the only significant single contrast at $p=0.10$ ($0.32°$ against $0.58°$, $p=0.03$), the pooled difference not significant ($p=0.19$), and the largest mean gap at $p=0.40$ smaller than its own seed-to-seed spread. In the i.i.d. model the highest-energy points are preferentially superpositions (Table 16), so raising their weight pulls the cluster centres toward mixed directions — the opposite of what happens on real speech. The repair this mechanism names, a weight on the *purity* of a point estimated blindly as its collinearity with the nearest cluster centre, was implemented and is reported as a tested negative in both domains: on the ladder it recovers part of the penalty ($2.49°\\to2.42°$ at the eighth power) but reaches the plain gate in no regime, and on real speech it is statistically indistinguishable from the collinearity weight it was meant to improve on — $0.676°$ against $0.677°$ at the second power, $0.704°$ and $0.689°$ at the fourth and eighth, none of the three paired contrasts significant ($p\\ge0.51$ over the $15\\times8$ cells). We therefore report the weighted variant as established on real speech and unsupported on the i.i.d. ladder — the purity weight included — rather than as a uniformly better method.'

A8_OLD = 'With the conventional reference scale, where the gate is effectively inactive ($96\\%$ of the points survive it), the weight carries almost the whole gain on its own: $8.26°\\to1.11°$.'

B3A_OLD = 'holding the max-referenced coefficient at $0.05$ costs $0.99°$ on average against $0.72°$ for the per-configuration optimum, a penalty of $0.27°$, while holding the conventional median-referenced default at\nits $0.02$ costs an order of magnitude against the $8.26°$ to $0.69°$ gap of Section 9.'

B3B_OLD = 'holding $c=0.05$ costs $0.99°$ on average against $0.72°$ for the per-configuration optimum, a penalty of $0.27°$, whereas holding the median-referenced default at its conventional $0.02$ costs an order of magnitude.'

B7_OLD = 'Its deficit against SCA-tuned is $0.08$–$0.19°$ over the four sparse regimes'

# ---------------------------------------------------------------- 修订条目
# (编号, 说明, old, new)
ENTRIES: list[tuple[str, str, str, str, str]] = [
    ('A1', '§9.5 删除逐字重复的一段（两份只差一处表格引用，删引 Table 16 的那份）',
     '\n\nFirst, the energy weight alone fails at SNR $0$ dB ($7.32°$, against $1.41°$ for the gate): when signal and noise energies are comparable the weight cannot separate them, and the hard removal performed by the gate is still required — it is the *combination* of gate and weight that we recommend. Second, the synthetic ladder gives no gain at all, and the sign there is the one the amplitude–purity relation predicts: the weighted mean is $2.49°$ against $2.12°$ for the plain gate, worse in four of the six regimes, with the only significant single contrast at $p=0.10$ ($0.32°$ against $0.58°$, $p=0.03$), the pooled difference not significant ($p=0.19$), and the largest mean gap at $p=0.40$ smaller than its own seed-to-seed spread. In the i.i.d. model the highest-energy points are preferentially superpositions (Table 16), so raising their weight pulls the cluster centres toward mixed directions — the opposite of what happens on real speech. The repair this mechanism names, a weight on the *purity* of a point estimated blindly as its collinearity with the nearest cluster centre, was implemented and is reported as a tested negative in both domains: on the ladder it recovers part of the penalty ($2.49°\\to2.42°$ at the eighth power) but reaches the plain gate in no regime, and on real speech it is statistically indistinguishable from the collinearity weight it was meant to improve on — $0.676°$ against $0.677°$ at the second power, $0.704°$ and $0.689°$ at the fourth and eighth, none of the three paired contrasts significant ($p\\ge0.51$ over the $15\\times8$ cells). We therefore report the weighted variant as established on real speech and unsupported on the i.i.d. ladder — the purity weight included — rather than as a uniformly better method.',
     '',
     'preferentially superpositions (Table 15)'),

    ('A2', '§2 真实语音系数区间 5→0.02（网格范围，与合成侧一致）',
     'from $0.1$ to $20$ on the synthetic ladder and from $5$ to $1000$ on real speech (Table 13)',
     'from $0.1$ to $20$ on the synthetic ladder and from $0.02$ to $1000$ on real speech (Table 13)',
     'from $0.1$ to $20$ on the synthetic ladder and from $0.02$ to $1000$ on real speech (Table 13)'),

    ('A3', 'A.18 把 71× 的房间从 small room 2 改成 large room 2',
     'by a factor of $71$ in the mildest case\n(small room 2) and by $1.5$ to $4$ orders of magnitude in the others',
     'by a factor of $71$ in the mildest case\n(large room 2) and by $1.5$ to $4$ orders of magnitude in the others',
     'by a factor of $71$ in the mildest case\n(large room 2) and by $1.5$ to $4$ orders of magnitude in the others'),

    ('A4a', 'A.18 banded floor 的附录号 A.14→A.16',
     'banded floor of Appendix A.14 as the repair rather than the guard',
     'banded floor of Appendix A.16 as the repair rather than the guard',
     'banded floor of Appendix A.16 as the repair rather than the guard'),

    ('A4b', 'A.25 banded variant 的附录号 A.14→A.16',
     'taken per band, which is the banded variant of Appendix A.14 —',
     'taken per band, which is the banded variant of Appendix A.16 —',
     'taken per band, which is the banded variant of Appendix A.16 —'),

    ('A5', 'A.26 表格引用 Table 12→Table 45',
     'Table 12 tabulates the three summaries.',
     'Table 45 tabulates the three summaries.',
     'Table 45 tabulates the three summaries.'),

    ('A6', '§8.4 N=3 的优势倍数 21→43',
     'at $p=0.05$ (factors of $21$, $23$, $21$ and $12$ for $N=3$–$6$)',
     'at $p=0.05$ (factors of $43$, $23$, $21$ and $12$ for $N=3$–$6$)',
     'at $p=0.05$ (factors of $43$, $23$, $21$ and $12$ for $N=3$–$6$)'),

    ('A7', '§9.5 常规门留存率 96%→99%',
     'where the gate is effectively inactive ($96\\%$ of the points survive it)',
     'where the gate is effectively inactive ($99\\%$ of the points survive it)',
     'where the gate is effectively inactive ($99\\%$ of the points survive it)'),

    ('A8', '§9.5 把无法从 Table 16 核对的 1.11° 改为止确的 1.15° 并说明口径',
     'the weight carries almost the whole gain on its own: $8.26°\\to1.11°$.',
     'the weight carries almost the whole gain on its own: the last column of Table 16 drops from $8.26°$ to $1.15°$, which is the *ungated* weight; applying the same weight to the conventional gate itself — the column Table 16 does not carry, and which the record reports as $1.11°$ (`r5_unified_weight.json`, `med0.02+w4`) — leaves the reading unchanged, and we give both numbers rather than the one that is not in the table.',
     'the weight carries almost the whole gain on its own: the last column of Table 16 drops from $8.26°$ to $1.15°$, which is the *ungated* weight; applying the same weight to the conventional gate itself — the column Table 16 does not carry, and which the record reports as $1.11°$ (`r5_unified_weight.json`, `med0.02+w4`) — leaves the reading unchanged, and we give both numbers rather than the one that is not in the table.'),

    ('A9', 'A.13 复混合极差的归属与容差',
     'The three columns agree with each other to within $2\\%$ at every regime, and with the semi-analytic form to\nwithin the Monte-Carlo error wherever the latter applies; the largest spread between the real and\nthe per-frequency complex case is $2.4\\%$, at $p=0.40$.',
     'The three columns agree with each other to within $2.5\\%$ at every regime, and with the semi-analytic form to\nwithin the Monte-Carlo error wherever the latter applies; the largest spread anywhere in the table is\n$2.4\\%$, at $p=0.40$, and it is between the real and the *fixed* complex matrix ($64.86$ against\n$63.33$) — the per-frequency complex case is within $0.5\\%$ of the real one at every regime.',
     'The three columns agree with each other to within $2.5\\%$ at every regime, and with the semi-analytic form to\nwithin the Monte-Carlo error wherever the latter applies; the largest spread anywhere in the table is\n$2.4\\%$, at $p=0.40$, and it is between the real and the *fixed* complex matrix ($64.86$ against\n$63.33$) — the per-frequency complex case is within $0.5\\%$ of the real one at every regime.'),

    ('A10', 'A.14 最优 α 的改进幅度 9%→21%',
     'over the four cells the best $\\alpha$ improves on\n$10^{-4}$ by $9\\%$ at the worst cell and $12\\%$ at $(0.10,0)$ dB',
     'over the four cells the best $\\alpha$ improves on\n$10^{-4}$ by $21\\%$ at the worst cell and $12\\%$ at $(0.10,0)$ dB',
     'over the four cells the best $\\alpha$ improves on\n$10^{-4}$ by $21\\%$ at the worst cell and $12\\%$ at $(0.10,0)$ dB'),

    ('A11', '附录导言的自持表范围 18–23→18–45',
     'Tables 18–23 are self-contained and are placed here, in\nAppendix A, for exactly that reason.',
     'Tables 18–45 are self-contained and are placed here, in\nAppendix A, for exactly that reason.',
     'Tables 18–45 are self-contained and are placed here, in\nAppendix A, for exactly that reason.'),

    ('A12', '附录导言删掉已失效的节号 8.6/8.8',
     'and A.26–A.27 the two synthetic studies moved from Sections 8.6 and 8.8; all seven were moved out of the main text so that the argument is not interrupted.',
     'and A.26–A.27 the two synthetic studies moved out of the main text in an earlier round; all seven were moved so that the argument is not interrupted.',
     'and A.26–A.27 the two synthetic studies moved out of the main text in an earlier round; all seven were moved so that the argument is not interrupted.'),

    ('A13', 'A.18 的 Section 10.5 悬空（§10.5 已随 §10 搬进 A.25；check_refs 的枚举解析抓到）',
     'Sections 9 and 10.5 use recorded speech; this section replaces the synthetic noise by a',
     'Sections 9 and A.25 use recorded speech; this section replaces the synthetic noise by a',
     'Sections 9 and A.25 use recorded speech'),
    ('B1', '§4.2 不再声称 4% 的代入误差落在仿真吻合度内',
     'the substitution changes $r(p)$ by at most $0.1\\%$ for $p\\le0.10$ and by $4\\%$ at $p=0.20$ — both inside the agreement with simulation reported below — and reaches $13\\%$ only at $p=0.40$, where the small-argument expansion has already ceased to apply.',
     'the substitution changes $r(p)$ by at most $0.1\\%$ for $p\\le0.10$ and by $4\\%$ at $p=0.20$ — the latter just outside the $1.5\\%$ agreement with simulation reported below, which is why the $p=0.20$ entry is the one place where the two are reported side by side rather than folded together — and reaches $13\\%$ only at $p=0.40$, where the small-argument expansion has already ceased to apply.',
     'the substitution changes $r(p)$ by at most $0.1\\%$ for $p\\le0.10$ and by $4\\%$ at $p=0.20$ — the latter just outside the $1.5\\%$ agreement with simulation reported below, which is why the $p=0.20$ entry is the one place where the two are reported side by side rather than folded together — and reaches $13\\%$ only at $p=0.40$, where the small-argument expansion has already ceased to apply.'),

    ('B2', '§8.1 补上第六个制度的区间',
     'the per-regime intervals, in the order of the regimes above, are $[+9.54,+20.48]$, $[+6.31,+19.09]$, $[+3.99,+17.33]$, $[-0.76,+9.04]$ and $[+0.08,+0.35]$, the fourth covering zero and matching the non-significant $p=0.088$.',
     'the per-regime intervals, in the order of the regimes above, are $[+9.54,+20.48]$, $[+6.31,+19.09]$, $[+3.99,+17.33]$, $[-0.76,+9.04]$, $[+0.08,+0.35]$ and $[-0.74,+0.29]$, the fourth and sixth covering zero and matching the non-significant $p=0.088$ and $p=0.345$.',
     'the per-regime intervals, in the order of the regimes above, are $[+9.54,+20.48]$, $[+6.31,+19.09]$, $[+3.99,+17.33]$, $[-0.76,+9.04]$, $[+0.08,+0.35]$ and $[-0.74,+0.29]$, the fourth and sixth covering zero and matching the non-significant $p=0.088$ and $p=0.345$.'),

    ('B3a', 'A.7 明说 0.99°/0.72° 的种子集',
     'holding the max-referenced coefficient at $0.05$ costs $0.99°$ on average against $0.72°$ for the per-configuration optimum, a penalty of $0.27°$, while holding the conventional median-referenced default at\nits $0.02$ costs an order of magnitude against the $8.26°$ to $0.69°$ gap of Section 9.',
     'holding the max-referenced coefficient at $0.05$ costs $0.99°$ on average against $0.72°$ for the per-configuration optimum, a penalty of $0.27°$ — both read off the two held-out tuning seeds of that record, which is why they differ from the $0.88°$ and $0.685°$ of Table 14, computed on the eight evaluation seeds — while holding the conventional median-referenced default at\nits $0.02$ costs an order of magnitude against the $8.26°$ to $0.69°$ gap of Section 9.',
     'holding the max-referenced coefficient at $0.05$ costs $0.99°$ on average against $0.72°$ for the per-configuration optimum, a penalty of $0.27°$ — both read off the two held-out tuning seeds of that record, which is why they differ from the $0.88°$ and $0.685°$ of Table 14, computed on the eight evaluation seeds — while holding the conventional median-referenced default at\nits $0.02$ costs an order of magnitude against the $8.26°$ to $0.69°$ gap of Section 9.'),

    ('B3b', '§11 同步说明 0.99° 的种子集',
     'holding $c=0.05$ costs $0.99°$ on average against $0.72°$ for the per-configuration optimum, a penalty of $0.27°$, whereas holding the median-referenced default at its conventional $0.02$ costs an order of magnitude.',
     'holding $c=0.05$ costs $0.99°$ on average against $0.72°$ for the per-configuration optimum, a penalty of $0.27°$ (measured on the two held-out tuning seeds of `real_grid.json`; on the eight evaluation seeds of Table 14 the same two figures are $0.88°$ and $0.685°$), whereas holding the median-referenced default at its conventional $0.02$ costs an order of magnitude.',
     'holding $c=0.05$ costs $0.99°$ on average against $0.72°$ for the per-configuration optimum, a penalty of $0.27°$ (measured on the two held-out tuning seeds of `real_grid.json`; on the eight evaluation seeds of Table 14 the same two figures are $0.88°$ and $0.685°$), whereas holding the median-referenced default at its conventional $0.02$ costs an order of magnitude.'),

    ('B4a', 'Table 7 题注标明是均值',
     '**Table 7.** Runtime per problem (seconds, one CPU core, 10 seeds).',
     '**Table 7.** Runtime per problem (seconds, one CPU core, mean of 10 seeds per regime; the medians are $0.169$ and $0.176$ s for NF-SSP and SCA-default).',
     '**Table 7.** Runtime per problem (seconds, one CPU core, mean of 10 seeds per regime; the medians are $0.169$ and $0.176$ s for NF-SSP and SCA-default).'),

    ('B4b', 'Table 22 题注说明不含数据生成、total 独立计时',
     '**Table 22.** Cost of the front end, single core, median over repetitions. The synthetic column uses the $2112$-point ladder; the real column uses the $\\mathrm{win}=1024$ configuration, $4.8\\times10^4$ TF points.',
     '**Table 22.** Cost of the front end, single core, median over repetitions, with the total timed independently of the stage rows. The synthetic column uses the $2112$-point ladder; the real column uses the $\\mathrm{win}=1024$ configuration, $4.8\\times10^4$ TF points. The stages therefore do not sum to the total exactly; the problem is generated before the clock starts, which is why the synthetic total is smaller than the $0.184$ s of Table 7.',
     '**Table 22.** Cost of the front end, single core, median over repetitions, with the total timed independently of the stage rows. The synthetic column uses the $2112$-point ladder; the real column uses the $\\mathrm{win}=1024$ configuration, $4.8\\times10^4$ TF points. The stages therefore do not sum to the total exactly; the problem is generated before the clock starts, which is why the synthetic total is smaller than the $0.184$ s of Table 7.'),

    ('B5', 'A.3 把 falling 改成 rising',
     '$17$ of the $270$ decisions are material errors at $0.035$, falling only slightly to $19$ at\n$0.05$.',
     '$17$ of the $270$ decisions are material errors at $0.035$, rising only slightly to $19$ at\n$0.05$.',
     '$17$ of the $270$ decisions are material errors at $0.035$, rising only slightly to $19$ at\n$0.05$.'),

    ('B6', '§5.3 dispersion 计数 13→12',
     'On real speech the picture reverses: $13$ of the $15$ configurations never exceed a dispersion of $0.401$ across eight seeds',
     'On real speech the picture reverses: $12$ of the $15$ configurations never exceed a dispersion of $0.401$ across eight seeds',
     'On real speech the picture reverses: $12$ of the $15$ configurations never exceed a dispersion of $0.401$ across eight seeds'),

    ('B7', '§8.1 对 oracle 的缺口下界 0.08→0.07',
     'Its deficit against SCA-tuned is $0.08$–$0.19°$ over the four sparse regimes',
     'Its deficit against SCA-tuned is $0.07$–$0.19°$ over the four sparse regimes',
     'Its deficit against SCA-tuned is $0.07$–$0.19°$ over the four sparse regimes'),

    ('B8a', '§9 明说 π0 与 r 的区间排除了两个 dense 配置',
     'On these recordings the noise-only fraction sits at $\\pi_0\\approx0.43$–$0.51$, on the favourable side of the knee, and the measured reference-scale ratio is only $1.9$–$2.2$;',
     'On these recordings the noise-only fraction sits at $\\pi_0\\approx0.43$–$0.51$ (excluding the two dense configurations, where it falls to $0.413$ and $0.368$), on the favourable side of the knee, and the measured reference-scale ratio is only $1.9$–$2.2$ ($2.40$ and $2.80$ in those two);',
     'On these recordings the noise-only fraction sits at $\\pi_0\\approx0.43$–$0.51$ (excluding the two dense configurations, where it falls to $0.413$ and $0.368$), on the favourable side of the knee, and the measured reference-scale ratio is only $1.9$–$2.2$ ($2.40$ and $2.80$ in those two);'),

    ('B8b', '§9.2 同步',
     'First, at the nominal SNR $\\pi_0$ is $0.43$–$0.51$, so real speech sits on the *favourable* side of the knee of Section 4.3: the median remains inside the noise component and $r$ is $1.9$–$2.2$,',
     'First, at the nominal SNR $\\pi_0$ is $0.43$–$0.51$ excluding the two dense configurations ($0.413$ and $0.368$; $2.40$ and $2.80$ for $r$), so real speech sits on the *favourable* side of the knee of Section 4.3: the median remains inside the noise component and $r$ is $1.9$–$2.2$,',
     'First, at the nominal SNR $\\pi_0$ is $0.43$–$0.51$ excluding the two dense configurations ($0.413$ and $0.368$; $2.40$ and $2.80$ for $r$), so real speech sits on the *favourable* side of the knee of Section 4.3: the median remains inside the noise component and $r$ is $1.9$–$2.2$,'),

    ('B10', '删掉 Figures 清单里 Fig. 10/11 之间孤立的分隔线',
     'which is the signature of a reference-scale defect rather than a noise problem. (`fig_heatmap.pdf`)\n---\n',
     'which is the signature of a reference-scale defect rather than a noise problem. (`fig_heatmap.pdf`)\n',
     'which is the signature of a reference-scale defect rather than a noise problem. (`fig_heatmap.pdf`)\n'),

    ('B11', 'Table 33 补 α=10^-2 列并改为升序',
     '| Cell | $\\hat\\sigma^2/\\sigma^2$ | $\\alpha=10^{-1}$ | $\\alpha=10^{-3}$ | $\\alpha=10^{-5}$ | $\\alpha=10^{-4}$ |\n|---|---|---|---|---|---|\n| $p=0.20$, SNR $0$ dB | 1.39 | 4.14% / 0.057 / 11.77 | 0.92% / 0.010 / 10.69 | 0.25% / 0.001 / 18.39 | 0.45% / 0.003 / 12.88 |\n| $p=0.20$, SNR $10$ dB | 1.59 | 14.58% / 0.276 / 2.13 | 12.25% / 0.229 / 1.66 | 10.21% / 0.185 / 1.42 | 11.20% / 0.207 / 1.51 |\n| $p=0.10$, SNR $0$ dB | 1.19 | 4.75% / 0.123 / 11.40 | 2.05% / 0.056 / 9.85 | 0.98% / 0.024 / 11.11 | 1.37% / 0.036 / 8.63 |\n| $p=0.40$, SNR $0$ dB | 1.60 | 2.89% / 0.020 / 12.57 | 0.29% / 0.001 / 16.10 | 0.04% / 0.000 / 23.00 | 0.11% / 0.000 / 20.58 |',
     '| Cell | $\\hat\\sigma^2/\\sigma^2$ | $\\alpha=10^{-1}$ | $\\alpha=10^{-2}$ | $\\alpha=10^{-3}$ | $\\alpha=10^{-4}$ | $\\alpha=10^{-5}$ |\n|---|---|---|---|---|---|---|\n| $p=20$, SNR $0$ dB | 1.39 | 4.14% / 0.057 / 11.77 | 1.89% / 0.024 / 10.15 | 0.92% / 0.010 / 10.69 | 0.45% / 0.003 / 12.88 | 0.25% / 0.001 / 18.39 |\n| $p=20$, SNR $10$ dB | 1.59 | 14.58% / 0.276 / 2.13 | 13.36% / 0.252 / 1.81 | 12.25% / 0.229 / 1.66 | 11.20% / 0.207 / 1.51 | 10.21% / 0.185 / 1.42 |\n| $p=10$, SNR $0$ dB | 1.19 | 4.75% / 0.123 / 11.40 | 3.09% / 0.085 / 7.63 | 2.05% / 0.056 / 9.85 | 1.37% / 0.036 / 8.63 | 0.98% / 0.024 / 11.11 |\n| $p=40$, SNR $0$ dB | 1.60 | 2.89% / 0.020 / 12.57 | 0.88% / 0.003 / 13.79 | 0.29% / 0.001 / 16.10 | 0.11% / 0.000 / 20.58 | 0.04% / 0.000 / 23.00 |',
     '| Cell | $\\hat\\sigma^2/\\sigma^2$ | $\\alpha=10^{-1}$ | $\\alpha=10^{-2}$ | $\\alpha=10^{-3}$ | $\\alpha=10^{-4}$ | $\\alpha=10^{-5}$ |\n|---|---|---|---|---|---|---|\n| $p=20$, SNR $0$ dB | 1.39 | 4.10% / 0.057 / 11.77 | 1.90% / 0.024 / 10.15 | 0.90% / 0.010 / 10.69 | 0.40% / 0.003 / 12.88 | 0.20% / 0.001 / 18.39 |\n| $p=20$, SNR $10$ dB | 1.59 | 14.60% / 0.276 / 2.13 | 13.40% / 0.252 / 1.81 | 12.20% / 0.229 / 1.66 | 11.20% / 0.207 / 1.51 | 10.20% / 0.185 / 1.42 |\n| $p=10$, SNR $0$ dB | 1.19 | 4.70% / 0.123 / 11.40 | 3.10% / 0.085 / 7.63 | 2.10% / 0.056 / 9.85 | 1.40% / 0.036 / 8.63 | 1.00% / 0.024 / 11.11 |\n| $p=40$, SNR $0$ dB | 1.60 | 2.90% / 0.020 / 12.57 | 0.90% / 0.003 / 13.79 | 0.30% / 0.001 / 16.10 | 0.10% / 0.000 / 20.58 | 0.00% / 0.000 / 23.00 |'),

]

# 修订后必须仍在场的字面探针（终检）：(说明, 文本)
PROBES = [
    ("§9.5 只剩一段 'First, the energy weight alone fails'", "First, the energy weight alone fails at SNR $0$ dB"),
    ("保留的那段引 Table 15", "preferentially superpositions (Table 15)"),
    ("§2 系数区间", "from $0.02$ to $1000$ on real speech"),
    ("A.18 房间名", "(large room 2)"),
    ("banded floor 指向 A.16", "banded floor of Appendix A.16"),
    ("A.26 表格号", "Table 45 tabulates the three summaries"),
    ("§8.4 倍数", "factors of $43$, $23$, $21$ and $12$"),
    ("§9.5 留存率", "($99\\%$ of the points survive it)"),
    ("§4.2 不再声称 4% 在吻合度内", "just outside the $1.5\\%$ agreement with simulation"),
    ("§8.1 第六条区间", "$[-0.74,+0.29]$"),
    ("A.7 种子集说明", "two held-out tuning seeds of that record"),
    ("Table 7 均值标注", "mean of 10 seeds per regime"),
    ("Table 22 口径说明", "with the total timed independently of the stage rows"),
    ("A.3 rising", "rising only slightly to $19$"),
    ("§5.3 dispersion 计数", "$12$ of the $15$ configurations never exceed"),
    ("§8.1 缺口下界", "Its deficit against SCA-tuned is $0.07$"),
    ("§9 dense 排除说明", "excluding the two dense configurations"),
    ("Figures 块无孤立分隔线", "(`fig_heatmap.pdf`)\n\n**Fig. 11.**"),
    ("Table 33 七列", "| $\\alpha=10^{-2}$ |"),
    ("A.18 的节号", "Sections 9 and A.25 use recorded speech"),
]

# 修订后不得再出现的旧文本
ANTI = [
    ("重复段的一份", "preferentially superpositions (Table 16)"),
    ("旧系数区间", "from $5$ to $1000$ on real speech"),
    ("旧房间名", "(small room 2) and by $1.5$ to $4$ orders"),
    ("旧附录号", "banded floor of Appendix A.14"),
    ("旧表格号", "Table 12 tabulates the three summaries"),
    ("旧倍数", "factors of $21$, $23$, $21$ and $12$"),
    ("旧留存率", "($96\\%$ of the points survive it)"),
    ("旧 1.11°", "$8.26\\u00b0\\to1.11\\u00b0$"),
    ("旧的 9%", "by $9\\%$ at the worst cell"),
    ("旧表范围", "Tables 18–23 are self-contained"),
    ("旧节号", "moved from Sections 8.6 and 8.8"),
    ("旧 falling", "falling only slightly to $19$"),
    ("旧 dispersion 计数", "$13$ of the $15$ configurations never exceed"),
    ("孤立的 ---", "(`fig_heatmap.pdf`)\n---\n"),
    ("旧的 10.5 节号", "Sections 9 and 10.5 use recorded speech"),
]


def main() -> int:
    dry = "--dry" in sys.argv
    with open(MS, encoding="utf-8") as fh:
        text = fh.read()

    print(f"手稿：{MS}")
    projected = text
    for tag, why, old, new, done in ENTRIES:
        if old in projected:
            projected = projected.replace(old, new, 1)
    changed = miss = 0
    for tag, why, old, new, done in ENTRIES:
        if old in text:
            # 已应用过再跑时 old 不在场；这里 old 还在，说明真要改（或 new==old 的空操作）
            if not dry:
                text = text.replace(old, new, 1)
            changed += 1
            print(f"  ✓ {tag} {why}")
        elif (new and new in text) or (done and done in text):
            print(f"  – {tag} {why}：已改")
        else:
            miss += 1
            print(f"  ✗ {tag} {why}：未命中")

    if not dry:
        with open(MS, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"\n已落地：改 {changed} 条 | 未命中 {miss}")
    else:
        print(f"\n演练：将改 {changed} 条 | 未命中 {miss}")

    # 终检
    print("\n终检：")
    bad = 0
    for why, probe in PROBES:
        if probe not in projected:
            print(f"  ✗ 缺失：{why}")
            bad += 1
    for why, probe in ANTI:
        if probe in projected:
            print(f"  ✗ 仍在：{why}")
            bad += 1
    if bad:
        print(f"终检失败：{bad} 项")
        return 1
    print("终检通过：修订全部在场、旧文本全部退场")

    if not dry:
        r = subprocess.run([sys.executable, os.path.join(HERE, "check_refs.py")],
                           cwd=ROOT, capture_output=True, text=True)
        tail = [l for l in r.stdout.split("\n") if l.strip()][-1:] or [r.stderr.strip()]
        print("后置 check_refs：" + tail[0])
        if r.returncode:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
