# paper/ 目录说明

**顶层只有一份成品 PDF：`manuscript_cssp.pdf`（投稿版）。** 别的那几份都删了。

```
manuscript.md              唯一稿件源（Markdown）。改稿只改这里。
manuscript_cssp.pdf        唯一成品 PDF —— 要投的就是这一份。
cssp_title_page.md         Title page 源（投 Editorial Manager 时转 PDF）
cssp_title_page_notes.md   Title page 的内部说明（评审模式、待办）—— 别外传
cover_letter_cssp.md       投稿信源
data_licenses.md           数据与许可清单（给老师/编辑看）

submission/                上传包 —— 投 Editorial Manager 就传这 4 个文件
  README.txt               上传对照表（哪个系统项对应哪个文件）
  manuscript.pdf           = manuscript_cssp.pdf 的副本
  title_page.pdf / cover_letter.pdf
  manuscript_source.zip    LaTeX 源 + Fig1..Fig11（期刊强制要求，缺了不退审）

submission_latex/          投稿用 LaTeX 源包（tex + 11 张图 + 编译 PDF）

reviews/                   历史评审材料（审稿报告、逐轮回复、三份 triage）—— 保密
springer_template/         Springer 官方模板（本机 TeX Live 缺依赖，没用上）
review/                    通用/IEEE 格式版（带行号，给老师批注用）—— 只有跑了
                           `python3 src/build_pdf.py` 才会出现
```

## 更新稿件的正确顺序

```bash
python3 src/build_pdf_cssp.py --repo https://github.com/BinNong/reference-scale-ubss
python3 src/pack_submission.py
```

第一条重建 `manuscript_cssp.pdf`，第二条刷新 `submission/` 上传包。
跑完顶层仍然只有一份稿件 PDF。

## 2026-10-16 整理前的问题

顶层同时躺着四份稿件 PDF，其中两份是陈旧构建：

| 文件 | 状态 |
|---|---|
| `main.pdf`（9/19） | 陈旧构建，p8 还是 Proposition 6 —— 已删 |
| `manuscript_sn_jnl.pdf`（10/7） | 陈旧构建，p8 是 Proposition 7 —— 已删 |
| `manuscript_submission.pdf`（10/15） | 与投稿版同内容、只差文献格式，重复 —— 已删 |
| `manuscript_prev_mdde.md`（9/14） | 更早的稿件草稿 —— 已删 |

删除前的完整备份在 `/tmp/paper_backup_20261016/`。

`build_pdf.py` 的默认输出也因此改到 `paper/review/`：顶层永远只有
`manuscript_cssp.pdf` 一份成品，不会再出现"两份 PDF 分不清投哪份"。
