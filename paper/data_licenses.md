# 数据集与许可清单（投稿合规核查）

> 目的：回答"用了哪些数据集、会不会侵权"。许可条款取自各语料的 **OpenSLR 官方页面**
> （2026-10-02 于服务器实测 `curl https://www.openslr.org/<N>/`，非二手转述）。
> 本文件不是法律意见，是投稿合规核查记录；正式声明已写入稿件 `## Declarations` 小节。

## 一、实际使用的数据

| # | 数据 | OpenSLR | 许可（实测） | 用在哪 | 稿中引用 |
|---|---|---|---|---|---|
| 1 | **LibriSpeech dev-clean**（16 kHz 朗读语音，1382 条） | SLR12 | **CC BY 4.0** | §9 真实语音评测（960 runs）、§10.5、贡献 5 | **[26]**（已有） |
| 2 | **Nigerian English**（众包语音，31 说话人 / 3359 条 / 48 kHz，Lagos 与 London 录制） | SLR70 | **CC BY-SA 4.0** | §9.6 / §A.20 / Table 45（960 runs） | **[29]**（本轮新增） |
| 3 | **REVERB 2014 房间脉冲响应与噪声**（8 元阵列、六房间，16 kHz） | SLR28 | **Apache 2.0** | §10.5 / §A.18 / Table 43 | **[30]**（本轮新增） |
| 4 | 合成数据（Monte Carlo，数万次随机配置） | — | 自生成，无第三方权利 | §1–§8 全部合成实验 | — |

**未使用**：代码里出现的 `zenodo.org` / `openairlib.net` 只是第九轮的**网络可达性探测与下载失败记录**，
没有采用其数据；`simulated_rirs/`（受控 T60）也未进入稿件。

## 二、逐条合规判断

### 1. LibriSpeech — CC BY 4.0
允许任何目的（含商业）使用、再分发、改编，**唯一义务是署名**。稿件已引用 [26]。
语音内容层面：LibriSpeech 源自 **LibriVox 公共领域有声书**（Panayotov 等，ICASSP 2015），
录音本身无版权。→ **不侵权**。

### 2. REVERB 2014 RIR/噪声（SLR28）— Apache 2.0
允许任何用途，义务是**保留版权与许可声明（NOTICE）**并附免责条款；**无 ShareAlike**，
比 CC 系更宽松。稿件只做评测、不再分发语料。→ **不侵权**。

### 3. Nigerian English（SLR70）— CC BY-SA 4.0 ⚠️ 唯一需要留意的一条
允许研究/商业使用与再分发，义务有二：**署名** + **相同方式共享（ShareAlike）**。

- ShareAlike **只在分发"改编材料"（adapted material）时触发**。
- 本稿与配套仓库**不含语料本身**（只给下载脚本/路径），**不构成改编材料的分发** → **SA 不触发**。
- ⚠️ **若将来把处理后的音频**（截取的 3 s 片段、混合后的 wav、示例输入/输出）**放进补充材料或公开仓库**，
  那些文件就是 SLR70 的衍生材料，**必须整体以 CC BY-SA 4.0 发布并署名**，且与 MIT/Apache 代码许可**不兼容**。
  → **建议：只发脚本、配置与记录，不发音频样本**；确要发，则把该部分单独以 CC BY-SA 4.0 标注。
- **隐私**：SLR70 是志愿者众包录音，发布者已按 CC 条款取得开放许可；稿件只报聚合角度误差、
  不发布可识别个人的音频。→ 无额外隐私问题。

## 三、稿件里已做的合规动作（R10e）

`src/apply_r10e_declarations.py` 在 §12 之后新增 **`## Declarations`**（无编号小节），含五项：

1. **Data availability**：点名三个语料 + OpenSLR 编号 + 许可 + 引用号，并**明写
   "No corpus is redistributed with this paper"**（这句让 SLR70 的 SA 义务"显然不触发"，可查验）。
2. **Code availability**：指向代码/记录仓库（地址待填）。
3. **Ethics**：只用公开且已获许可的语料，未采集新的人类参与者数据，无需伦理审批。
4. **Competing interests**：无。
5. **Funding**：待填。

参考文献新增 **[29] SLR70**、**[30] SLR28** 两条（署名是许可义务，不只是学术惯例）。

## 四、仓库与投稿状态（2026-10-03）

**代码与记录仓库（已发布、公开）：**

> **https://github.com/BinNong/reference-scale-ubss**

仓库含 **200 个文件 / 8.6 MB**：`src/`(107) `results/`(59：48 个记录 + 11 张矢量图)
`legacy/`(25) `scripts/`(4)，加 `README.md`、`requirements.txt`、`.gitignore`、
`.gitattributes`、本文件。**不含语料、不含手稿、不含评审材料**——与稿件中
"No corpus is redistributed with this paper" 的声明一致。

**已处理完的投稿前项**：

| 项 | 状态 |
|---|---|
| 仓库地址 | ✅ 已注入 §7 与 Declarations（构建时用 `--repo <URL>`） |
| 基金信息 | ✅ 手稿已写 "no funds, grants, or other support were received" |
| 评审模式 | ✅ **单盲**（single-anonymous）→ 正文**保留**作者信息，不做匿名化 |

> 构建命令（两个成品都要带 `--repo`）：
> ```bash
> python3 src/build_pdf.py      --repo https://github.com/BinNong/reference-scale-ubss
> python3 src/build_pdf_cssp.py --repo https://github.com/BinNong/reference-scale-ubss
> ```
> 两个构建器都会在成品里回查 `[repository to be inserted]`、`[funding information to be inserted]`、
> `[add campus/city if required]` 等草稿痕迹，命中即以非零码退出——**有意设计**，
> 防止占位符悄悄留在投稿版里。
