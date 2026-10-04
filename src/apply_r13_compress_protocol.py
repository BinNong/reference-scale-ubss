r"""R13：把 §7 的逐个归档清单压成分类汇总，改指向仓库 README。

为什么做这一处
==============
CSSP 第十轮正式意见 3.7：

    59 页仍偏长，建议正文 30–35 页

当时的实测是正文 36 页 / 附录 18 页；现在正文 **42 页** / 附录 20 页 / 共 62 页
（第十一轮加了 §10.4 与 Table 22、A.19 与 Table 44、A.20 与 Table 45），
离意见建议的区间超出 7–12 页。意见点名可压的三样是
「实验记录清单、robustness bookkeeping、重复的 caveat」，其中第一样是
`paper/cssp_review_triage.md` §5 记录里**唯一被判"不删任何论断"**的那处，
且至今未做。本脚本只动它。

改什么
======
§7「Implementation and reproducibility」是一段 544 词的单行，其中约 400 词是
逐个 json 文件名 + 记录数 + 轮次归属的罗列。这些信息在仓库 README 的
*Records → scripts* 表里全有，那才是它们的维护位置。压成分类汇总后：

* **保留全部论断**（一条不删）：耗时与三个对比、Table 27 的分 stage 代价、
  "一个自包含脚本 + 固定种子集"、§4–5 派生量由两个脚本独立复核并复现
  $p_c$（0.2%）、"每张表都有归档"、可用性声明与仓库地址、`requirements.txt`；
* 记录数改为**分组总数**；
* 顺带修掉一处算错的合计：原文写 "Appendix A adds a further $1{,}320$
  records"，可它自己列举的七项是 240+300+270+48+144+192+150 = **1,344**，
  差 24 —— 正是同段后文宣布的 `r2_noise_family.json` 由 120 扩到 144
  那一下，合计从未跟着改。逐个列举才会养出这种不一致；
* 可用性声明由"上文点名的每个归档"改为"全部归档 + README 逐个点名"——
  后者仍是可查验的断言，且断言对象（README 在仓库里）比正文里的枚举更难失配。

连带登记（"登记而非删除"）
==========================
这一段是另外两个**活跃**脚本的锚。删掉枚举会让它们从"跳过"退化成
"未命中 / 退出 1"，即**新增未命中**，违反"每轮不得新增未命中"的判据：

  1. `apply_r7b_fig_renumber.py`
     `ANCHOR = "gate-plus-weight design of Section 9.5"`，
     `edit_sentence` 命中 0 次时主流程 `die(...)`。
     → 登记：锚缺失时记 0 次并放行；锚仍在时判据不变（必须恰好 1 次）。

  2. `apply_r8c_manifest.py`
     两条条目的 `old` 都在这一段内，`old not in s` 时 `sys.exit(1)`。
     → 登记：两行整表登记为"已登记并跳过"。

`verify_r10.py` 那条探针的**判据不变**（它作用于全文，而 §A.20 自己就点名
`r10_real_slr70.json` 与 `r10_real_slr70_grid.json`），只把标签改成如实描述
记录名现在的位置——标签说谎比断言宽松更难发现。

用法：
    python3 src/apply_r13_compress_protocol.py --dry
    python3 src/apply_r13_compress_protocol.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"

# ---------------------------------------------------------------- 新正文
NEW = (
    r"**Implementation and reproducibility.** The estimator is a few hundred lines of "
    r"NumPy and SciPy with no training and no GPU requirement; it runs in $0.184$ s per "
    r"problem on one CPU core — essentially the same as the conventional pipeline "
    r"($0.189$ s) and $7.7\times$ faster than the smoothed-$\ell_0$ baseline ($1.41$ s) "
    r"(Table 7). Table 27 breaks the cost down by stage, which is the form in which the "
    r"claim that the calibration is nearly free can be checked. All experiments are "
    r"driven by one self-contained script from a fixed seed set, and the derived "
    r"quantities of Sections 4–5 are verified independently by two further scripts that "
    r"evaluate the regularised incomplete Beta and chi-square functions directly and "
    r"reproduce $p_c$ by Monte Carlo, agreeing with the derivation to within $0.2\%$."
    "\n\n"
    r"**Records.** Every table is backed by a machine-readable record in `results/`, and "
    r"every record is produced by exactly one script from a fixed seed set. The synthetic "
    r"ladder and its sweeps account for $2{,}770$ runs and the real-speech suite for "
    r"$960$; the energy- and purity-weighted variants add $768$ and $1{,}392$, the "
    r"scaling study $54$ cells, and Tables 1–4 each have a dedicated closed-form or "
    r"self-check record. A robustness suite covering the quality of the admitted set, "
    r"the sensitivity of $\eta$ and $q_{\max}$, the noise families, the mixing geometry "
    r"and the sparsity–SNR plane adds $1{,}344$ records. Successive revisions add "
    r"density front ends and a robust-scale comparison, head-to-head density "
    r"clustering, the complex- and convolutive-mixing study, the $\alpha$ sensitivity, "
    r"the banded floor, order selection, the gate-plus-weight design of Section 9.5, "
    r"the source-number mismatch study, the recorded room, an eigenstructure sweep of "
    r"$4{,}400$ column subsets and a second corpus of $960$ runs. The repository README "
    r"names every record and maps each table and figure to the script and record that "
    r"produce it, so any number in this paper can be regenerated from the release alone; "
    r"code, the fixed configurations, the dependency list and all records are available "
    r"at <https://github.com/BinNong/reference-scale-ubss>, together with a "
    r"`requirements.txt` pinning the environment."
)

# 段落的起点标记与紧随其后的空行（该段在稿中是单行，故 "\n\n" 即段落边界）
START = "**Implementation and reproducibility.**"

# 改动后必须仍在的论断（终检用）
KEEP = [
    r"$0.184$ s per problem on one CPU core",
    r"($0.189$ s)",
    r"$7.7\times$ faster than the smoothed-$\ell_0$ baseline ($1.41$ s)",
    "Table 27 breaks the cost down by stage",
    "one self-contained script from a fixed seed set",
    r"agreeing with the derivation to within $0.2\%$",
    "Every table is backed by a machine-readable record",
    "https://github.com/BinNong/reference-scale-ubss",
    r"`requirements.txt` pinning the environment",
    r"$2{,}770$ runs",
    r"$960$",
    r"$768$ and $1{,}392$",
    "$54$ cells",
    r"Tables 1–4 each have a dedicated closed-form or self-check record",
    r"$1{,}344$ records",
    r"$\eta$ and $q_{\max}$",
    r"sparsity–SNR plane",
    r"$4{,}400$ column subsets",
    "gate-plus-weight design of Section 9.5",
]
# 逐个枚举的痕迹，改后必须消失
GONE = [
    "`nfr_results.json`",
    "`r2_ssp_quality.json`",
    "`r2_eta_sweep.json`",
    "`r10_real_slr70_grid.json`",
    "named above",
    "$1{,}320$ records",
]

# 两个被本脚本解除锚定的活跃脚本，及其补丁（幂等标记 -> 补丁内容）
R7B = ROOT / "src" / "apply_r7b_fig_renumber.py"
R8C = ROOT / "src" / "apply_r8c_manifest.py"
V10 = ROOT / "src" / "verify_r10.py"

R7B_OLD = (
    "    body, n_edit = edit_sentence(ms, direction, old_no, new_no)\n"
    "    if n_edit != 1:\n"
    "        die(f\"§7 那句没被处理（edit 命中 {n_edit} 次）——核对 {ANCHOR!r} 的现行形态\")\n"
)
R7B_NEW = (
    "    body, n_edit = edit_sentence(ms, direction, old_no, new_no)\n"
    "    # R13（apply_r13_compress_protocol.py）把 §7 的逐个归档清单换成分类汇总，\n"
    "    # ANCHOR 那句连带消失。按\"登记而非删除\"的约定登记：反向重放无法把它插回，\n"
    "    # 故锚缺失时记 0 次并放行，不让往返测试死在这句上。锚仍在时判据不变。\n"
    "    if n_edit != 1 and ANCHOR in ms:\n"
    "        die(f\"§7 那句没被处理（edit 命中 {n_edit} 次）——核对 {ANCHOR!r} 的现行形态\")\n"
    "    if ANCHOR not in ms:\n"
    "        print(\"  [R13 已登记] §7 归档清单已压成分类汇总，ANCHOR 句不存在，\"\n"
    "              \"edit_sentence 记 0 次并放行\")\n"
)
R7B_MARK = "R13 已登记"

R8C_OLD = (
    "def main() -> None:\n"
    "    ap = argparse.ArgumentParser()\n"
    "    ap.add_argument(\"--dry\", action=\"store_true\")\n"
    "    a = ap.parse_args()\n"
)
R8C_NEW = (
    "# R13 把 §7 的逐个归档清单换成分类汇总，下面两条条目的锚文本随之消失。\n"
    "# 按\"登记而非删除\"的约定整表登记：稿件已不含这些锚，故两行都应\"已登记并跳过\"，\n"
    "# 而不是未命中退出。判据仍是\"不得新增未命中\"——登记是显式的，不是静默放过。\n"
    "SUPERSEDED_R13 = {\n"
    "    \"§7 清单：r2_noise_family 记录数 120 -> 144\",\n"
    "    \"§7 清单：补第八轮归档 r8_n_mismatch.json\",\n"
    "}\n"
    "\n"
    "\n"
    "def main() -> None:\n"
    "    ap = argparse.ArgumentParser()\n"
    "    ap.add_argument(\"--dry\", action=\"store_true\")\n"
    "    a = ap.parse_args()\n"
)
R8C_MARK = "SUPERSEDED_R13"

R8C_LOOP_OLD = (
    "        if old not in s:\n"
    "            print(f\"✗ 未命中：{tag}\")\n"
    "            sys.exit(1)\n"
)
R8C_LOOP_NEW = (
    "        if old not in s:\n"
    "            if tag in SUPERSEDED_R13:\n"
    "                skipped += 1\n"
    "                print(f\"  [R13 已登记，跳过] {tag}\")\n"
    "                continue\n"
    "            print(f\"✗ 未命中：{tag}\")\n"
    "            sys.exit(1)\n"
)

V10_OLD = (
    "        want(\"§A.20 与 §7 清单点名两个归档\",\n"
    "             \"r10_real_slr70.json\" in t and \"r10_real_slr70_grid.json\" in t)\n"
)
V10_NEW = (
    "        # R13 起 §7 的归档清单压成分类汇总，两个记录名只在 §A.20 出现；\n"
    "        # 判据不变（作用于全文），标签如实反映位置。\n"
    "        want(\"§A.20 点名两个归档（§7 清单 R13 已压成分类汇总）\",\n"
    "             \"r10_real_slr70.json\" in t and \"r10_real_slr70_grid.json\" in t)\n"
)

# ---------------------------------------------------------------- 另外三个
# 这三个活跃 applier 里各有一条"往 §7 清单里点名新归档"的条目，锚定的正是被压掉的
# 那段枚举句。它们的结构各不相同（两个是 `n == 0` 分支 + missing 计数，一个是
# 直接 die），故分别登记；共同点是"判据仍要求不得新增未命中"，登记是显式的。
R10B = ROOT / "src" / "apply_r10b_gram_spectrum.py"
R10D = ROOT / "src" / "apply_r10d_second_corpus.py"
R8D = ROOT / "src" / "apply_r8d_realroom.py"

# apply_r10b / r10d：结构相同，逐字复用同一对补丁
_ENUM_OLD = (
    "            print(f\"  ✗ 未命中（原文不存在）：{tag}\")\n"
    "            missing += 1\n"
    "            continue\n"
)
_ENUM_NEW = (
    "            # R13（apply_r13_compress_protocol.py）把 §7 的归档清单压成分类汇总，\n"
    "            # 本条锚定的 §7 枚举句随之消失。按\"登记而非删除\"的约定登记：\n"
    "            # 条目效果已在稿中，只是不再由本脚本追加。\n"
    "            if tag in SUPERSEDED_R13:\n"
    "                skipped += 1\n"
    "                print(f\"  [R13 已登记，跳过] {tag}\")\n"
    "                continue\n"
    "            print(f\"  ✗ 未命中（原文不存在）：{tag}\")\n"
    "            missing += 1\n"
    "            continue\n"
)


def _enum_def(path: pathlib.Path) -> tuple[str, str]:
    """按目标文件的 E 表生成"插入 SUPERSEDED_R13"的补丁对。

    tag **从脚本里抓**（AST 解析 `E`，取 tag 以 `§7` 开头的那条），不在本文件里
    重抄一遍：转写错一个汉字就会"登记了一个不存在的 tag"，脚本照旧未命中退出，
    而登记看起来是打上了——正是最难发现的那类假绿。抓到 0 条或多条都直接报错。

    插入型补丁必须原样保留锚点（`E = [` 那行不能动），故 old 包含在 new 中——
    幂等由标记保证，调用方传 preserve_anchor=True。
    """
    import ast
    tree = ast.parse(path.read_text())
    tags: list[str] = []
    for node in ast.walk(tree):
        # `E = [...]` 是 Assign；`E: list[...] = [...]` 是 AnnAssign——两种都要认，
        # 否则带类型标注的文件会被判成"0 条"，把登记 silently 打空。
        if isinstance(node, ast.Assign):
            tgt = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            tgt = node.target
        else:
            continue
        if getattr(tgt, "id", "") != "E":
            continue
        for el in node.value.elts:
            tags.append(ast.literal_eval(el.elts[0]))
    hits = [x for x in tags if x.startswith("§7")]
    if len(hits) != 1:
        die(f"{path.name}：tag 以 §7 开头的条目有 {len(hits)} 条（期望 1）——"
            f"抓到 {hits}")
    tag = hits[0]
    anchor = "E = [\n" if "E = [" in path.read_text() \
        else "E: list[tuple[str, str, str]] = [\n"
    if path.read_text().count(anchor) != 1:
        die(f"{path.name}：E 定义锚点不唯一")
    new = (
        "# R13 把 §7 的逐个归档清单换成分类汇总，下面锚定 §7 枚举句的条目随之失效。\n"
        "# 按\"登记而非删除\"的约定登记；判据仍是\"不得新增未命中\"。\n"
        "SUPERSEDED_R13 = {\n"
        f"    {tag!r},\n"
        "}\n"
        "\n"
        "\n" + anchor
    )
    return anchor, new


R8D_OLD = (
    "        if old not in s or s.count(old) != 1:\n"
    "            die(f\"「{tag}」：旧文出现 {s.count(old)} 次（要求唯一）\")\n"
)
R8D_NEW = (
    "        if old not in s or s.count(old) != 1:\n"
    "            # R13：§7 枚举句被压成分类汇总，本条随之失效，显式登记后跳过。\n"
    "            if old not in s and tag in SUPERSEDED_R13:\n"
    "                skipped += 1\n"
    "                print(f\"  [R13 已登记，跳过] {tag}\")\n"
    "                continue\n"
    "            die(f\"「{tag}」：旧文出现 {s.count(old)} 次（要求唯一）\")\n"
)

# ---------------------------------------------------------------- 两个 verifier
# 压缩让"每个归档都要在正文点名"这条纪律换了承载对象：论文现在说的是
# "README names every record and maps each table and figure to the script and
# record that produce it"。纪律不放松，只是断言对象改成论文真正指向的那份文件。
#
# verify_revision 那条 $1{,}320$ 探针要单独说：它**把错的那个数字钉住了**。
# 七个归档实有条数之和是 1,344，稿里一直写 1,320（差 24 = `r2_noise_family.json`
# 由 120 扩到 144 那一下、合计从未跟着改），而探针只检查"1{,}320 出现在正文里"，
# 于是错得越久越"通过"。R13 把它换成从稿中解析该合计、与七个归档之和对齐。
VR = ROOT / "src" / "verify_revision.py"
VR_PROP_OLD = '                    "Robustness and Cost", "1{,}320"]:'
VR_PROP_NEW = '                    "Robustness and Cost", "1{,}344"]:'
VR_SUM_OLD = (
    '    check("  scaling 54", 54, '
    'len(json.loads((R / "scale_invariance.json").read_text())["rows"]), 0.0)\n'
)
VR_SUM_NEW = VR_SUM_OLD + (
    "\n"
    "    # R13 起 §7 的逐个枚举压成分组总数，\"鲁棒性套件\"那组只给一个合计。\n"
    "    # 判据随之改为**从手稿解析该合计、与七个归档的实条数对齐**。\n"
    "    # 早先这里是写死的字面量探针 \"1{,}320\"，只查\"这个数出现在正文里\"，\n"
    "    # 而七个归档之和其实是 1,344（差 24 = r2_noise_family 由 120 扩到 144 那一下、\n"
    "    # 合计从未跟着改）——探针正好把错数字钉住，错得越久越\"通过\"。\n"
    "    _rb = [\"r2_ssp_quality.json\", \"r2_ssp_quality_real.json\", \"r2_eta_sweep.json\",\n"
    "           \"r2_qmax_sweep.json\", \"r2_noise_family.json\", \"r2_angle_sweep.json\",\n"
    "           \"r2_heatmap_p_snr.json\"]\n"
    "    _tot = sum(len(json.loads((R / _n).read_text())[\"records\"]) for _n in _rb)\n"
    "    print(f\"  robustness 套件七个归档合计 = {_tot}\")\n"
    "    if ms is not None:\n"
    "        _m = re.search(r\"adds \\$([^$]+)\\$ records\", ms)\n"
    "        if not _m:\n"
    "            print(\"  FAIL §7 Records 段解析不出鲁棒性套件合计\")\n"
    "            FAIL.append(\"§7 robustness 合计可解析\")\n"
    "        else:\n"
    "            check(\"  §7 robustness 套件合计\",\n"
    "                  int(re.sub(r\"[^0-9]\", \"\", _m.group(1))), _tot, 0.0)\n"
)

# verify_r8：两条 manifest 探针改查 README
R8V = ROOT / "src" / "verify_r8.py"
R8V_FLAT_OLD = "    flat = re.sub(r\"\\s+\", \" \", text)\n"
R8V_FLAT_NEW = (
    "    flat = re.sub(r\"\\s+\", \" \", text)\n"
    "    # R13：§7 改由 README 逐个点名归档，故读它。缺失时给空串——判据应当失败，\n"
    "    # 而不是因为\"文件不在\"被跳过。\n"
    "    readme = ((ROOT / \"README.md\").read_text()\n"
    "              if (ROOT / \"README.md\").exists() else \"\")\n"
)
R8V_OLD_1 = '    want("manifest 点名 r8_n_mismatch.json", "r8_n_mismatch.json" in flat)\n'
R8V_NEW_1 = (
    "    # R13 起 §7 不再逐个点名归档（压成分组总数，改由仓库 README 点名并映射\n"
    "    # 表/图 -> 脚本 -> 归档）。纪律不放松，只把断言对象换成论文指向的那份文件。\n"
    "    want(\"README 点名 r8_n_mismatch.json\", \"r8_n_mismatch.json\" in readme)\n"
)
R8V_OLD_2 = '    want("manifest 点名 r8_real_room.json", "r8_real_room.json" in flat)\n'
R8V_NEW_2 = '    want("README 点名 r8_real_room.json", "r8_real_room.json" in readme)\n'


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def sec7(text: str) -> str:
    i, j = text.index("## 7. Experimental Protocol"), text.index("## 8. Results")
    return text[i:j]


def para_words(text: str) -> int:
    return len(text.split())


def patch(srcs: dict, path: pathlib.Path, old: str, new: str, mark: str, tag: str,
          dry: bool, preserve_anchor: bool = False) -> None:
    """幂等打补丁：已含标记则跳过；old 唯一则替换；否则报错。

    补丁后的源码留在 `srcs` 里而不是立刻落盘，这样 dry-run 的终检查的是
    **将要写入的状态**而不是磁盘上的旧状态——后者会让 dry-run 假失败。

    `preserve_anchor`：**插入型**补丁要把 `old` 原样留在 `new` 里（锚点所在的
    `def main()` 头不能动），此时 `old in new` 恒为真。这条护栏的本意是防"文稿
    文本改完还会再改一次"，而这里的幂等由 `mark` 保证——标记在就不碰文件，
    比 old/new 不相交更强。故对插入型补丁显式关掉该检查，其余补丁照旧检查。
    """
    s = srcs[path]
    if mark in s:
        print(f"  [已登记，跳过] {tag}")
        return
    n = s.count(old)
    if n != 1:
        die(f"{path.name}：补丁锚点出现 {n} 次（期望 1）——{tag}")
    if not preserve_anchor and old in new:
        die(f"{path.name}：{tag} 的 new 包含完整 old，会破坏幂等")
    srcs[path] = s.replace(old, new, 1)
    print(f"  ✓ {tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--manuscript", default=None)
    a = ap.parse_args()
    ms_path = pathlib.Path(a.manuscript) if a.manuscript else MS
    s = ms_path.read_text()

    print("=== 1. §7 归档清单：逐个枚举 -> 分类汇总 ===")
    if NEW.split("\n\n", 1)[1] in s:
        print("  [已改，跳过] 新文本已在稿中")
    else:
        head = sec7(s)
        if head.count(START) != 1:
            die(f"§7 里起点标记出现 {head.count(START)} 次（期望 1）")
        k = s.index(START)
        end = s.index("\n\n", k)
        old_para = s[k:end]
        # 防转写错误：对被替换的那一段做性质断言
        if not old_para.endswith("pinning the environment."):
            die("被替换段不是预期的归档清单段（结尾不符）")
        if para_words(old_para) != 544:
            die(f"被替换段词数 {para_words(old_para)}，期望 544——段边界可能已变")
        if "$1{,}320$" not in old_para or "`nfr_results.json`" not in old_para:
            die("被替换段缺少预期的枚举内容")
        print(f"  被替换段：{len(old_para)} 字符 / {para_words(old_para)} 词")
        s = s[:k] + NEW + s[end:]
        print(f"  新文本：{len(NEW)} 字符 / {para_words(NEW)} 词"
              f"（−{para_words(old_para) - para_words(NEW)} 词）")

    if s.count("$") % 2:
        die(f"$ 不配对：{s.count('$')}")

    print("\n=== 2. 连带登记：解除另外五个活跃脚本的锚 ===")
    # 补丁后的源码先留在内存里，落盘推迟到终检通过之后
    TARGETS = (R7B, R8C, V10, R10B, R10D, R8D, VR, R8V)
    srcs = {p: p.read_text() for p in TARGETS}
    patch(srcs, R7B, R7B_OLD, R7B_NEW, R7B_MARK,
          "apply_r7b_fig_renumber.py：ANCHOR 缺失时放行", a.dry)
    patch(srcs, R8C, R8C_OLD, R8C_NEW, R8C_MARK,
          "apply_r8c_manifest.py：登记 SUPERSEDED_R13 表", a.dry, preserve_anchor=True)
    patch(srcs, R8C, R8C_LOOP_OLD, R8C_LOOP_NEW, "SUPERSEDED_R13:",
          "apply_r8c_manifest.py：命中登记表即跳过而非退出", a.dry)
    patch(srcs, V10, V10_OLD, V10_NEW, "§7 清单 R13 已压成分类汇总",
          "verify_r10.py：标签改为如实描述记录名位置", a.dry)
    for p in (R10B, R10D):
        d_old, d_new = _enum_def(p)
        tag = ("apply_r10b_gram_spectrum.py" if p is R10B
               else "apply_r10d_second_corpus.py")
        patch(srcs, p, d_old, d_new, "SUPERSEDED_R13",
              f"{tag}：登记 SUPERSEDED_R13 表", a.dry, preserve_anchor=True)
        patch(srcs, p, _ENUM_OLD, _ENUM_NEW, "SUPERSEDED_R13:",
              f"{tag}：命中登记表即跳过而非计入未命中", a.dry, preserve_anchor=True)
    patch(srcs, R8D, *_enum_def(R8D), "SUPERSEDED_R13",
          "apply_r8d_realroom.py：登记 SUPERSEDED_R13 表", a.dry, preserve_anchor=True)
    patch(srcs, R8D, R8D_OLD, R8D_NEW, "SUPERSEDED_R13:",
          "apply_r8d_realroom.py：命中登记表即跳过而非 die", a.dry, preserve_anchor=True)
    patch(srcs, VR, VR_PROP_OLD, VR_PROP_NEW, "R13",
          "verify_revision.py：1{,}320 探针 -> 1{,}344", a.dry)
    patch(srcs, VR, VR_SUM_OLD, VR_SUM_NEW, "R13 起 §7",
          "verify_revision.py：合计改为从七个归档现算并与稿中解析值对齐", a.dry,
          preserve_anchor=True)
    patch(srcs, R8V, R8V_FLAT_OLD, R8V_FLAT_NEW, "R13：§7 改由 README",
          "verify_r8.py：读入 README", a.dry, preserve_anchor=True)
    patch(srcs, R8V, R8V_OLD_1, R8V_NEW_1, "README 点名 r8_n_mismatch.json",
          "verify_r8.py：r8_n_mismatch 改查 README", a.dry)
    patch(srcs, R8V, R8V_OLD_2, R8V_NEW_2, "README 点名 r8_real_room.json",
          "verify_r8.py：r8_real_room 改查 README", a.dry)

    if a.dry:
        print("\n[dry] 未写入。")
    else:
        ms_path.write_text(s)
        print(f"\n已写入 {ms_path}")
        for p in TARGETS:
            if srcs[p] != p.read_text():
                p.write_text(srcs[p])
                print(f"已写入 {p.relative_to(ROOT)}")

    # ------------------------------------------------------------ 终检
    # 注意：查的是**将要写入的状态**（内存里的 s / srcs），不是磁盘。
    # dry-run 时磁盘还是旧稿，查磁盘会让终检假失败、把"改对了"误报成"改错了"。
    print("\n=== 终检 ===")
    t = s
    bad = [p for p in KEEP if p not in t]
    for p in KEEP:
        print(f"  {'OK  ' if p in t else 'FAIL'} 论断仍在：{p[:58]}")
    if bad:
        die(f"终检失败：{len(bad)} 条论断丢失")
    # 枚举痕迹只应在 §7 内消失。`r10_real_slr70_grid.json` 等名字在 §A.20
    # 里是**故意保留**的——verify_r10 那条探针判据作用于全文，靠的就是它们。
    for g in GONE:
        ok = g not in sec7(t)
        print(f"  {'OK  ' if ok else 'FAIL'} §7 内枚举痕迹已消失：{g[:40]!r}")
        if not ok:
            die(f"终检失败：§7 内枚举痕迹仍在 {g!r}")
    if "$1{,}320$" not in sec7(t) and "$1{,}320$" in t:
        die("终检失败：算错的合计挪了位置但没改")
    if "$1{,}320$" in sec7(t):
        die("终检失败：算错的合计仍在 §7")
    for p, m in ((R7B, R7B_MARK), (R8C, R8C_MARK), (V10, "已压成分类汇总"),
                 (R10B, "SUPERSEDED_R13"), (R10D, "SUPERSEDED_R13"),
                 (R8D, "SUPERSEDED_R13"), (VR, "R13 起 §7"),
                 (R8V, "README 点名 r8_n_mismatch.json")):
        ok = m in srcs[p]
        print(f"  {'OK  ' if ok else 'FAIL'} 已登记：{p.name}")
        if not ok:
            die(f"终检失败：{p.name} 未登记")
    # 被登记的脚本必须仍是合法 Python，且判据本身没被改坏
    import py_compile
    for p in (R7B, R8C, V10, R10B, R10D, R8D, VR, R8V):
        try:
            py_compile.compile(str(p), doraise=True, cfile="/tmp/_r13_pyc.pyc")
            print(f"  OK   {p.name} 语法有效")
        except py_compile.PyCompileError as e:
            die(f"终检失败：{p.name} 语法错误：{e}")
    print("\n✓ 全部终检通过")


if __name__ == "__main__":
    main()
