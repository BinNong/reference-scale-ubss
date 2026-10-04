# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code, fixed configurations and machine-readable records for the paper *"The Reference
Scale of Energy Thresholding in Sparse-Domain Blind Source Separation"*. The repo is **public**
(github.com/BinNong/reference-scale-ubss) and its availability statement promises exactly one
thing: code + fixed configurations + records. Everything is CPU-only, single-process,
fixed-seed.

Two consequences that are easy to get wrong:

- **`paper/manuscript.md`, the referee reports, `response_to_review_*.md`, the triage notes, the
  cover letter, the title page and `paper/submission*/` are confidential and gitignored.** They
  exist locally, they are not in the public repo, and a whole family of scripts reads them.
  Never commit them.
- **Nothing identifying may enter a committed file** — no author names, emails, affiliations,
  local usernames or absolute home paths. Gate every change with
  `git grep -nIE 'nong[b]in|tfswuf[e]|foxmai[l]|Bin [N]ong|/User[s]/' -- .` (must be empty).
  The brackets are deliberate: this file *states* the rule, so it cannot spell the pattern
  out — a literal pattern would flag this very line. The gate still bites everywhere else.
  Commit identity stays neutral (`reference-scale-ubss <…@users.noreply.github.com>`).

## Environment

```bash
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

The thread variables are **not optional**: the runtime table is a per-core measurement, and an
oversubscribed BLAS pool silently ruins both timings and worker pools.

**Experiments run from `src/`** and write into `../results/`. All modules use flat imports
(`import data as D`), so there is no package to install.

Local vs. remote: `/usr/local/bin/python3` here has numpy 2.5 / scipy 1.18 / matplotlib, but
**no scikit-learn**. Every script that clusters (`baselines.kmeans_sphere`) therefore only runs
on the compute host, where the venv sits at the **repository root** (`../.venv/bin/python` when
called from `src/`). Pure-NumPy scripts (`r10_gram_spectrum.py`, `r4_floor_bias.py`, …) do run
locally. Before comparing any local result with a remote record, check the md5 of
`data.py / metrics.py / baselines.py / nfr.py` match.

## Commands

### Regenerate one record

Every record in `results/` is produced by exactly one script; the README's *Records → scripts*
table is the mapping. Taking a record therefore means one command:

```bash
cd src
python <script>.py --out ../results/<record>.json          # e.g. python validate_guard.py --out ../results/guard_validation.json
python r2_sweeps.py --part eta --seeds 10 --out ../results/r2_eta_sweep.json
```

Records are the **only** source of every number in the manuscript. After any re-run, push the
record to the compute host **and** confirm with `./scripts/check_records_sync.sh` (exit 2 means
"could not reach the host", not "identical"). A record left behind after a re-run is the failure
mode this exists for.

### Run a suite

```bash
python experiments_nfr.py --out ../results/nfr_results.json                          # ~tens of minutes
python experiments_real.py --out ../results/real_results.json --grid-out ../results/real_grid.json
python experiments_real.py --corpus <SLR70> --out ../results/r10_real_slr70.json --grid-out ../results/r10_real_slr70_grid.json
bash r2_run.sh      # robustness suite; honours PYTHON= and W=, logs to ../logs/
bash r4_run.sh      # Tables 27–30 + the front-end study
bash r5_run.sh      # Tables 31–39, Fig. 11; sets R5_NO_SDR=1
```

`r4_run.sh` and `r5_run.sh` hardcode `../.venv/bin/python` and (for `r5`) a server-side `cd`,
so they only work as written on the compute host.

Real speech needs the corpus, which is not redistributed: `../data/ls_corpus/**/*.flac`
(LibriSpeech dev-clean), passed to `realdata.scan_corpus()`.

### Worker pools

Must use `mp.get_context("spawn")`. The parent initialises BLAS before forking, and a forked
child then inherits a locked mutex and blocks forever — it looks like a hang on an idle machine.
`spawn` does **not** inherit module-level globals, so pools that need them use
`ProcessPoolExecutor(initializer=...)`.

### The post-edit loop

Run after every manuscript edit. Appliers run from the **repository root**; `verify_*` and
`check_tables` can be run from anywhere (they resolve the repo root themselves), except
`verify_revision.py` which uses the relative path `../paper/manuscript.md` and must be run from
`src/`.

```bash
for f in src/apply_*.py; do python3 "$f" --dry; done          # applier idempotence
python3 src/check_tables.py paper/manuscript.md               # every "mean" row vs. its column
cd src
for v in verify_revision.py verify_r4.py verify_r5.py verify_r6.py verify_r7.py verify_r8.py verify_r9.py verify_r10.py; do python3 $v; done
```

`verify_*.py` **parse the manuscript for literal values** rather than hardcoding them, and each
supports `--manuscript <path>` for negative testing — a verifier that cannot fail is worthless,
so prove it bites. `validate_guard.py`, `validate_nfr.py` and `r2_stats.py` need only the
records and can be run without the manuscript at all.

### Build the PDF

```bash
python3 src/build_pdf.py --repo <URL>           # generic review PDF
python3 src/build_pdf_cssp.py --repo <URL>      # CSSP submission PDF (anonymised + Springer refs)
python3 src/pack_submission.py                  # rebuilds both, then the LaTeX + upload bundles
```

`build_pdf.py` needs `pandoc` + `xelatex`, and **it re-checks the finished PDF**: table and
figure numbering (asserting both the set and the landing order), section numbers read from the
`.aux`, equation order, abstract claims, bibliography count and leftover placeholders. Compiling
is not the same as being correct; this step has caught several silent errors. Without `--repo`
the PDF is still written but the script **exits 2** on the `[repository to be inserted]`
placeholder. `build_pdf_cssp.py` derives from the markdown and never writes it.

### Figures

```bash
python make_figures_nfr.py                      # Figs. 1–6
python regen_fig.py architecture guard          # regenerate *only* these
python3 test_fig_geometry.py                    # the repo's only test: the diagram self-check
```

Regenerate individual figures, never `main()` — a full run rewrites every figure and makes it
impossible to tell which one you changed. Confirm with `cmp` afterwards.

### Remote host

`scripts/rssh.sh '<cmd>'` / `scripts/rsync.sh` read the connection profile from
`remote_server.md` (gitignored, holds credentials). They fall back to `scripts/rssh_py.py`
(paramiko) when the local OpenSSH client is unusable — the symptom is `No user exists for uid N`,
which is a local passwd problem, nothing to do with the network or the server.

## Architecture

### The pipeline, and why `nfr.py` is only a mask

The paper's two stages are detection and clustering; in code there are three steps:

```
gate (which TF points are single-source)  →  clusterer (estimate A)  →  debiased ℓ1 recovery (estimate S)
```

The contribution is entirely in the **gate's** reference scale. `nfr.nfr_mask()`
returns a boolean mask plus a diagnostics dict — it is not a method and owns no clusterer. It
drops into any existing pipeline: `experiments_nfr.e6()` swaps it into K-means / FCM /
potential-function / SL0 front ends unchanged. `baselines.py` supplies the clusterers and the
recoverer, and `nfr.run_nfr()` wires the three together.

The three criteria, deliberately isomorphic to the classical pipeline so that only the reference
scale differs: single-source `|cos(X_r, X_i)| > c₀`, real/imag balance, and energy. The energy
one is `e > τ·ν̂` where `ν̂ = M·ŝ²` is a **blind noise floor** and `τ` is closed form
(`threshold_ratio`). Two ideas carry the paper:

1. `estimate_noise_power()` recovers `s²` from the exact lower tail — for small quantiles
   `s² = 2·e_(q)/Q_q(χ²_{2M})` holds exactly, so a run of order statistics gives a run of `s²`
   estimates whose median is the answer. A small-`x` power-law fit is *not* used: it is only
   accurate for `q ≲ 4e-4` at M=2 and biases high by ~60%.
2. `threshold_ratio(M, α) = Q_{1-α}(χ²_{2M})/(2M)` depends on **nothing but (M, α)** — so
   "tuning the threshold" becomes a closed-form calibration instead of a per-regime hyperparameter
   search. That is the paper's central claim.

The self-check matters as much as the estimator: if the fitted spread is too wide, or the gate
would retain almost nothing, the criterion decides the premise "noise-separable from signal" has
failed and **degrades honestly to the classical pipeline** instead of failing silently.

The disease it treats is in `ssp_mask_from_complex()` (`baselines.py`), which uses
`e > t_e · median(e)`: the median drifts with sparsity by roughly two orders of magnitude, so any
fixed `t_e` is wrong at one end and breaks silently.

### Module roles

| Module | Role |
|---|---|
| `data.py` | synthetic TF-domain UBSS problems; the complex→real equivalence (`tf_to_real`) is exact, not an approximation |
| `baselines.py` | the six comparators + sphere K-means / FCM / potential-function / SL0 / DUET + FISTA-LASSO-with-debiasing recovery. All methods share the recoverer, so comparisons are fair |
| `metrics.py` | BSS_EVAL-style SDR/SIR/SAR with Hungarian matching, and the mixing-matrix angle error the tables report |
| `nfr.py` | the gate: reference scale, blind noise floor, closed-form τ, self-check, `run_nfr()` |
| `realdata.py` | LibriSpeech construction; mixes in the time domain then STFTs, which is exactly per-column frequency-domain mixing |
| `experiments_nfr.py` / `experiments_real.py` | the two drivers; every record comes from one of these or a per-round script |

### The record contract

`results/*.json` is the single source of truth for every number. Scripts are one-shot and
parameter-explicit; they never read a record to decide what to write. The round structure is
visible in the filenames: a later round typically adds `apply_rN_*.py` (manuscript edit),
`verify_rN.py` (check the edit) and `results/rN_*.json` (new experiments). Some records are
explicitly superseded (`r5_complexity.json` by `r6_quantile_cost.json`) and stay only for
provenance.

`results/` also holds records from abandoned lines — `mdde_*`, `energy_weight`,
`fairness_check`, `real_mechanism`, `theory_validation`. They source no table or figure; do not
quote them.

### Code that is not part of the paper

- `src/model.py` + `src/experiments_mdde.py` — the abandoned SA-DUN deep-unfolding line. It
  needs PyTorch, which is **not** in `requirements.txt` and is not installed. Leave it alone.
- `legacy/` — the archive of the first positioning, kept for comparison only.
- `results/figures_v2/` — **a decoy**: same filenames, different figures. Never glob
  `results/figures_*/fig_*.pdf`; the manuscript uses `figures_nfr/` and `figures_real/` only.

## Traps

- **Appliers are one-shot manuscript editors, not build steps.** `apply_*.py` hold large
  literal `old`/`new` text blocks. Early-round ones are **frozen** — their `__main__` prints
  `[R8 已冻结]` and writes nothing, because a later round renumbered every table and appendix
  subsection and replaying them writes stale numbers (or re-inserts sentences). Active ones are
  `apply_r7b_fig_renumber.py`, `apply_r8*`, `apply_r9*`, `apply_r10*`. Idempotence is two-tier:
  frozen scripts must print the freeze notice and not write; active scripts must report
  "all skipped, zero misses" — and the criterion is *no new misses*, not zero. They must be run
  from the repository root (relative path `paper/manuscript.md`).
- **`new` must never contain the whole of `old`** in an applier entry, or the entry fires again
  on every run. Superseded entries are *registered* (`_SUPERSEDED_*` / `_HAND_CORRECTED`), never
  deleted — deleting them hides that the edit ever happened.
- **Never index by a number that gets renumbered.** Table and figure numbers have been reshuffled
  more than once (`verify_r5/r6/r7` cope with a `_R8_TAB` shim that re-hangs tables under their old
  numbers; `build_pdf.FIGWIDTH` is keyed by *filename* for the same reason). Key by filename, key
  or path.
- **Assert counts derived from the manuscript, never hardcoded.** Every "expect N" assertion in
  `build_pdf.py` and the verifiers derives N from the source; hardcoded expectations have passed
  while the content was wrong.
- **Equation numbers live in the build layer, not the manuscript.** The markdown still says
  `$$…$$`; `build_pdf.number_equations()` rewrites `\[`/`\]` into numbered environments and counts
  from the `$$` pairs. Do not go looking for `(1)` in `paper/manuscript.md`.
- LaTeX written inside an applier must be a **raw string** — a bare `\a` collapses to BEL(0x07)
  and LaTeX rejects it as an invalid character.
- BSD `grep` does not support `\|`; use `grep -nE`. And in zsh, a `ls a* b*` where one glob fails
  aborts the whole command.
- `.workbuddy/memory/` (gitignored) holds the round-by-round history of every review cycle in
  Chinese, including the reasoning behind each decision. It is the deepest context available
  and is worth reading before a structural change.
