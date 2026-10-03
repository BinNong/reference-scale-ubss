# Reference scale of energy thresholding in sparse-domain UBSS

Code, fixed configurations and machine-readable records for

> **The Reference Scale of Energy Thresholding in Sparse-Domain Blind Source Separation:
> A Diagnosis and a Blind Self-Calibrating Gate**

The study asks why the energy gate that every two-stage UBSS pipeline applies before
clustering is so sensitive to source sparsity, shows that the conventional *reference
scale* (a fixed multiple of the median, or of the maximum, of the observed energies) drifts
with sparsity by nearly two orders of magnitude, and replaces it with a gate referenced to
a blindly estimated noise floor and calibrated once to a false-admission rate.

Nothing here needs a GPU. Every experiment is a fixed-seed, single-process computation;
the estimator itself is a few hundred lines of NumPy and SciPy.

---

## Environment

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the reported numbers (Python 3.12,
NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.6.1, Matplotlib 3.10.3, SoundFile 0.13.1).

**Set the thread count before running anything.** The experiments are single-threaded by
design — the runtime table (Table 7) is a per-core measurement — and a stray BLAS thread
pool both distorts it and, more importantly, makes a worker pool oversubscribe the machine:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

All commands below are run **from `src/`**, and write into `../results/`.

---

## Layout

```
src/                all experiment, verification and plotting scripts
results/            machine-readable records (*.json) and vector figures (*.pdf)
scripts/            optional wrappers for running on a remote host
requirements.txt    the pinned environment
```

The manuscript, the referee reports and the point-by-point responses are **not** part of this
repository. What is published here is the code, the fixed configurations, the machine-readable
records and this README -- the same set the manuscript's availability statement refers to.
Nothing else is needed to regenerate every reported number.

`scripts/rssh.sh` and `scripts/rsync.sh` are the author's remote-runner wrappers; they read
a connection profile from `remote_server.md` and are **not** needed to reproduce anything.
Every script below runs locally with the same arguments.

If the local OpenSSH client is unusable — the symptom is `No user exists for uid N`, which means
the local user has no passwd entry and OpenSSH aborts before contacting any host — both wrappers
fall back automatically to `scripts/rssh_py.py`, a paramiko channel with `run` / `push` / `pull`.
That fallback needs paramiko in the Python that runs the wrappers:

```bash
# only if the local ssh client is broken
pip install --trusted-host mirrors.aliyun.com paramiko
```

`scripts/check_records_sync.sh` compares `results/` against the compute host by MD5 and needs the
same channel; it is the check that catches a record left behind after a re-run.

### Data

The synthetic experiments generate their own data (`data.py`) and need nothing.

The real-speech experiments read a LibriSpeech `dev-clean` corpus. Download it separately
and point the loader at it — no audio is redistributed here:

```bash
# expected layout
data/ls_corpus/**/*.flac        # LibriSpeech dev-clean, unmodified
```

`realdata.scan_corpus()` takes the corpus root as an argument, so the placement is free as
long as you pass the right path (the scripts default to `../data/ls_corpus`).

---

## Reproducing the results

### Records → scripts

Every record cited in the manuscript is produced by exactly one script. Run the script and
you regenerate the record; the record is in turn the sole source of the numbers in the
tables.

| Record (`results/`) | Script (`src/`) | Produces |
|---|---|---|
| `nfr_results.json` | `experiments_nfr.py` | synthetic ladder, SNR / sensor / source sweeps, α sweep, plug-in front ends, runtimes (Tables 6–13, 27) |
| `real_results.json` | `experiments_real.py` | 15 real-speech configurations × 8 seeds (Tables 14–17) |
| `prop6_convention.json` | `prop6_convention.py` | the energy convention of Proposition 6 and the closed form vs measurement (Tables 1–2) |
| `prop6_orthogonality.json` | `prop6_orthogonality.py` | the orthogonality approximation, quantified |
| `guard_validation.json` | `validate_guard.py` | 27-condition retention self-check (Tables 4, 25) |
| `synth_tuned_grid.json` | `synth_tuned_grid.py` | per-regime tuned-threshold bound, synthetic (Tables 15, 18) |
| `scale_invariance.json` | `scale_invariance.py` | $1/\sqrt{FT}$ scaling study (Fig. 7, §8.8) |
| `weighted_variant.json` | `weighted_variant.py` | energy-weighted variant, 15 real + synthetic ladder (Tables 17–18) |
| `purity_stage2.json` | `purity_weight.py --stage 2` | purity-weighted variant, real speech (§9.5) |
| `r2_ssp_quality.json` | `r2_ssp_quality.py` | mask-level precision / recall / $F_1$ / rejection rates, synthetic (Table 23) |
| `r2_ssp_quality_real.json` | `r2_ssp_quality_real.py` | the same at mask level, real speech (Table 24) |
| `r2_eta_sweep.json` | `r2_sweeps.py --part eta` | retention-threshold sensitivity (Table 25) |
| `r2_qmax_sweep.json` | `r2_sweeps.py --part qmax` | lower-tail-cap sensitivity (Table 26) |
| `r2_angle_sweep.json` | `r2_sweeps.py --part angle` | minimum mixing-angle sweep (Table 20) |
| `r2_noise_family.json` | `r2_noise_family.py` | Gaussian / Laplacian / impulsive / colored / uniform noise (Table 19) |
| `r2_heatmap_p_snr.json` | `r2_heatmap.py` | sparsity × SNR plane (Fig. 11) |
| `r2_runtime.json` | `r2_runtime.py` | component-wise runtime (Table 27) |
| `r2_stats.json` | `r2_stats.py` | confidence intervals, $d_z$, Holm and Benjamini–Hochberg (Table 28) |
| `r2_prop6_rank.json` | `r2_prop6_rank.py` | where the Proposition 6 approximation comes from: activity rank, not column angle (§4.2) |
| `r4_guard_errors.json` | `r4_guard_errors.py` | per-seed retention decisions, their location, and the dispersion-condition ablation (Table 29, §5.3) |
| `r4_density_frontend.json` | `r4_density_frontend.py` | density pre-screening ahead of the gate, three radii (Table 30) |
| `r4_robust_floor.json` | `r4_robust_floor.py` | MAD / 25th-percentile / lower-tail floor estimators, synthetic and real (Table 31) |
| `r4_table2.json` | `r4_table2.py` | the estimator's bias against sparsity, with the seed sensitivity of the dense row (Table 2) |
| `r4_floor_bias.json` | `r4_floor_bias.py` | finite-sample bias of the level convention, pure noise, $n^{-1/2}$ decay (§5.1) |
| `r5_density_baseline.json` | `r5_density_baseline.py` | DBSCAN / OPTICS / density-peak as head-to-head competitors, synthetic (Table 33) |
| `r5_density_real.json` | `r5_density_baseline.py --real` | the same on the real-speech subset (Table 34) |
| `r5_complex.json` | `r5_complex.py` | reference-scale ratio, criterion quality and gate behaviour under complex mixing (Tables 35–37) |
| `r5_alpha_cell.json` | `r5_alpha_cell.py` | the tolerance where the floor estimate is biased (Table 38) |
| `r5_complexity.json` | `r5_complexity.py` | the quantile step against $n=FT$, full sort and a histogram approximation, **as first measured** -- superseded by `r6_quantile_cost.json` |
| `r5_band_floor.json` | `r5_band_floor.py` | per-band noise floor, and why it is not the repair for babble (Table 40) |
| `r5_source_number.json` | `r5_source_number.py` | gate strictness and automatic order selection (Table 21) |
| `r5_unified_weight.json` | `r5_unified_weight.py` | the gate-plus-weight design: three rules × weighted/unweighted (Table 17, §9.5, Fig. 10) |
| `r6_quantile_cost.json` | `r6_quantile_cost.py` | the quantile step against $n=FT$, **re-measured** -- the record behind Table 39 and Appendix A.15 |
| `r8_n_mismatch.json` | `r8_n_mismatch.py` | source-number mismatch, on the ladder and on real speech (Table 22, §10.4) |
| `r8_real_room.json` | `r8_real_room.py` | recorded room noise, six rooms of the 2014 REVERB challenge (Table 43, §10.5, Appendix A.18) |
| `r10_gram_spectrum.json` | `r10_gram_spectrum.py` | sensitivity to the eigenstructure of the active Gram (Table 44, Appendix A.19) |
| `r10_real_slr70.json` | `experiments_real.py --corpus <SLR70> --grid-out ...` | the second corpus, under the protocol of §9.1 (Table 45, Appendix A.20); `r10_real_slr70_grid.json` holds its tuning grids |

Two further variants are driven by `weighted_variant.py` (`weighted_stage1.json`) and
`purity_weight.py --stage 1` (`purity_stage1.json`); the stage-1 records are the reduced
grids used during development and are **superseded** by the full runs above.

> **Table and figure numbers refer to the version of the manuscript this release accompanies.**
>
> `results/` also contains `mdde_results.json`, `mdde_ablation.json`, `energy_weight.json`,
> `fairness_check.json`, `theory_validation.json`, `real_grid.json` and
> `real_mechanism.json`. These belong to earlier or auxiliary lines of work -- an unfolded
> network that was abandoned, an energy-weighting pilot, and the mechanism study behind
> Fig. 8c. They are kept for provenance and **are not sources of any table or figure** in the
> present manuscript except where noted above.

### Figures → scripts

| Figure | File (`results/figures_nfr/`, `results/figures_real/`) | Script |
|---|---|---|
| Fig. 1 reference-scale ratio $r(p)$ | `fig_scale.pdf` | `make_figures_nfr.py` |
| Fig. 2 retention diagnostic | `fig_guard.pdf` | `make_figures_nfr.py` |
| Fig. 3 pipeline | `fig_architecture.pdf` | `make_figures_nfr.py` (or `regen_fig.py architecture`) |
| Fig. 4 sparsity | `fig_sparsity.pdf` | `make_figures_nfr.py` |
| Fig. 5 SNR | `fig_snr.pdf` | `make_figures_nfr.py` |
| Fig. 6 source count | `fig_nsources.pdf` | `make_figures_nfr.py` |
| Fig. 7 scaling | `fig_scaling.pdf` | `make_fig_scaling.py` |
| Fig. 8–9 real speech | `fig_real_scale.pdf`, `fig_real_main.pdf` | `make_figures_real.py` |
| Fig. 10 real speech distilled | `fig_real_summary.pdf` | `r5_fig_real_summary.py` |
| Fig. 11 sparsity × SNR | `fig_heatmap.pdf` | `r2_heatmap.py` |

### One-shot reproduction

```bash
cd src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiments_nfr.py --out ../results/nfr_results.json      # ~ tens of minutes
python experiments_real.py --out ../results/real_results.json    # needs the corpus
bash r2_run.sh                                                   # the robustness suite (Tables 19–26, Fig. 10)
bash r4_run.sh                                                   # Tables 27–29 and the front-end study
bash r5_run.sh                                                   # Tables 31–39, Fig. 11
python make_figures_nfr.py                                       # Figs. 1–6
```

There are three batch wrappers — `r2_run.sh`, `r4_run.sh`, `r5_run.sh` — one per round of
robustness experiments; each writes its own log under `logs/` on the host that runs it.

---

## Verifying rather than trusting

Several scripts check the records -- and the manuscript built from them -- rather than the
other way round. They are cheap and are the ones to run after any edit.

The `verify_*.py` and `check_tables.py` checks read `paper/manuscript.md`, which is **not**
part of this release. They are included so that a reader who has the manuscript can re-run
them verbatim; every check that needs only the records runs from `results/` alone.

| Script | What it checks |
|---|---|
| `verify_revision.py` | every number quoted in the text against its record; no external libraries beyond NumPy/SciPy |
| `check_tables.py` | the `mean` row of every table against the column it summarises |
| `r2_stats.py` | the statistical claims (intervals, effect sizes, multiplicity) |
| `verify_r4.py` | every number in the Appendix-A tables added in the fourth round |
| `verify_r5.py` | the fifth-round tables, **read out of the manuscript itself** rather than from constants, including two cross-table checks (Table 17 must agree with Table 15 column by column, and with the records behind it); `--manuscript <path>` supports negative testing |

Two more scripts check the manuscript *structure*:

| Script | What it checks |
|---|---|
| `test_fig_geometry.py` | that no arrow in the pipeline diagram passes through a box (a silent failure: it renders fine) |
| `build_pdf.py` | builds `paper/manuscript_submission.pdf` **and** re-checks the finished PDF for table/figure numbering, section numbers (read from the `.aux`), abstract claims, bibliography count and leftover placeholders |

`build_pdf.py` is also how the PDF is produced — it needs `pandoc` and `xelatex`:

```bash
python build_pdf.py                  # -> ../paper/manuscript_submission.pdf
python build_pdf.py --repo <URL>     # inserts the code/data repository address
python build_pdf.py --campus <text>  # appends campus/city to the affiliation
```

Without `--repo` the finished PDF still carries `[repository to be inserted]`, and the
build says so explicitly instead of passing.

---

## Notes that save time

- **Worker pools must use `spawn`.** The parent process evaluates the synthetic problems
  before it forks, which initialises the BLAS thread pool; a `fork`-ed child then inherits a
  locked mutex and blocks forever. It looks like a hang with the machine idle. All
  multiprocessing entry points pass `mp_context=mp.get_context("spawn")`.
- **Never fight an ambiguous `.venv`.** `experiments_real.py` defaults to six workers; with
  the thread-count variables unset the records become contention-limited (a median of 17.4 s
  per problem against 2.35 s measured cleanly on one core). The manuscript quotes the clean
  figures.
- **Real-speech runtimes in old records are not authoritative** for the same reason; use
  `r2_runtime.py`, which measures each stage separately on a single core.
